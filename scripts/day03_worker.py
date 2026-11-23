"""One process run of one Day 3 measurement phase on this chip's GPU, through MLX. Prints JSON.

  bandwidth  streaming kernels on 256 MiB arrays: scale, add, a fused triad, and a read-only sum
  matmul     square matrix multiply at three precisions and four sizes
  skinny     decode-shaped products: a batch of B vectors times one 8192 x 8192 fp16 matrix

Each configuration is first checked against a numpy reference, then timed in batches long enough to
hide the cost of launching work: the work is queued without waiting, one wait at the end. The run
keeps every per-call sample so the runner can take the fastest and the median.

usage: python scripts/day03_worker.py --phase bandwidth|matmul|skinny
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time

import mlx.core as mx
import numpy as np

from inferlab import roofline

N_STREAM = 64 * 1024 * 1024  # float32 elements: 256 MiB per array
SCALAR = 3.0
MATMUL_SIZES = (512, 1024, 2048, 4096)
MATMUL_DTYPES = {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}
BYTES_PER = {"fp32": 4, "fp16": 2, "bf16": 2}
SKINNY_K = SKINNY_N = 8192
SKINNY_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
SAMPLES = 8
MIN_BATCH_S = 0.05  # each sample queues enough calls to take at least this long


def wait_for_gpu() -> None:
    mx.synchronize()


def time_call(fn, samples: int = SAMPLES) -> list[float]:
    """Seconds per call, one entry per sample. Work is queued; there is one wait at the end."""
    out = fn()
    mx.eval(out)  # warm-up: compiles kernels and allocates
    start = time.perf_counter()
    out = fn()
    mx.eval(out)
    one_call = max(time.perf_counter() - start, 1e-6)
    calls = max(1, math.ceil(MIN_BATCH_S / one_call))
    result = []
    for _ in range(samples):
        wait_for_gpu()
        start = time.perf_counter()
        for _ in range(calls):
            out = fn()
            mx.async_eval(out)
        mx.eval(out)
        wait_for_gpu()
        result.append((time.perf_counter() - start) / calls)
    return result


def entry(flops: float, moved: float, samples: list[float], validated: bool, **extra) -> dict:
    return {
        "flops": flops,
        "bytes": moved,
        "best_s": min(samples),
        "median_s": statistics.median(samples),
        "samples_s": samples,
        "validated": validated,
        **extra,
    }


def close(got: np.ndarray, want: np.ndarray, tolerance: float) -> bool:
    """Relative L2 error of the whole vector, which does not depend on any one element's size."""
    denominator = float(np.linalg.norm(want))
    return denominator > 0 and float(np.linalg.norm(got - want)) / denominator <= tolerance


def phase_bandwidth() -> dict:
    mx.random.seed(0)
    a = mx.random.uniform(shape=(N_STREAM,))
    b = mx.random.uniform(shape=(N_STREAM,))
    mx.eval(a, b)
    head = 1 << 20
    na, nb = np.array(a[:head]), np.array(b[:head])

    @mx.compile
    def triad(x, y):
        return x + SCALAR * y

    kernels = {
        # name: (function, flops per element, bytes moved per element, reference on the head)
        "scale": (lambda: a * SCALAR, 1, 8, na * SCALAR),
        "add": (lambda: a + b, 1, 12, na + nb),
        "triad": (lambda: triad(a, b), 2, 12, na + SCALAR * nb),
    }
    configs = {}
    for name, (fn, flops_pe, bytes_pe, reference) in kernels.items():
        ok = close(np.array(fn()[:head]), reference, 1e-6)
        configs[name] = entry(
            flops_pe * N_STREAM, bytes_pe * N_STREAM, time_call(fn), ok, elements=N_STREAM
        )
    total = float(np.array(a).sum(dtype=np.float64))
    ok = abs(float(mx.sum(a)) - total) / total < 1e-4
    configs["sum"] = entry(N_STREAM, 4 * N_STREAM, time_call(lambda: mx.sum(a)), ok)
    return configs


def operands(m: int, k: int, n: int, dtype_name: str):
    scale = 1.0 / math.sqrt(k)
    x = (mx.random.normal((m, k)) * scale).astype(MATMUL_DTYPES[dtype_name])
    w = (mx.random.normal((k, n)) * scale).astype(MATMUL_DTYPES[dtype_name])
    mx.eval(x, w)
    return x, w


def host(array) -> np.ndarray:
    """A copy on the host in float32, which numpy can hold for every precision used here."""
    return np.array(array.astype(mx.float32))


def validated_product(x, w, dtype_name: str) -> bool:
    """The first row of x @ w against numpy, using the same rounded inputs."""
    tolerance = 1e-3 if dtype_name == "fp32" else 2e-2
    want = host(x[:1]) @ host(w)
    got = host((x @ w)[:1])
    return close(got, want, tolerance)


def phase_matmul() -> dict:
    mx.random.seed(0)
    configs = {}
    for dtype_name in MATMUL_DTYPES:
        width = BYTES_PER[dtype_name]
        for n in MATMUL_SIZES:
            x, w = operands(n, n, n, dtype_name)
            ok = validated_product(x, w, dtype_name)
            configs[f"{dtype_name}-{n}"] = entry(
                roofline.gemm_flops(n, n, n),
                roofline.gemm_bytes(n, n, n, width),
                time_call(lambda x=x, w=w: x @ w),
                ok,
                n=n,
                dtype=dtype_name,
            )
    return configs


def phase_skinny() -> dict:
    mx.random.seed(0)
    k, n = SKINNY_K, SKINNY_N
    w = (mx.random.normal((k, n)) / math.sqrt(k)).astype(mx.float16)
    mx.eval(w)
    configs = {}
    for batch in SKINNY_BATCHES:
        x = (mx.random.normal((batch, k)) / math.sqrt(k)).astype(mx.float16)
        mx.eval(x)
        ok = validated_product(x, w, "fp16")
        configs[f"batch-{batch}"] = entry(
            roofline.gemm_flops(batch, n, k),
            roofline.gemm_bytes(batch, n, k, 2),
            time_call(lambda x=x: x @ w),
            ok,
            batch=batch,
            k=k,
            n=n,
            dtype="fp16",
        )
    return configs


PHASES = {"bandwidth": phase_bandwidth, "matmul": phase_matmul, "skinny": phase_skinny}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    args = parser.parse_args()
    configs = PHASES[args.phase]()
    print(
        json.dumps(
            {
                "phase": args.phase,
                "mlx_version": getattr(mx, "__version__", None),
                "device": str(mx.default_device()),
                "configs": configs,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
