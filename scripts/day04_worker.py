"""One process run of one Day 4 phase: hand-written Metal kernels through MLX. Prints JSON.

  stream     the triad out = a + s * b on 64 Mi float32 elements, one element per thread (scalar)
             or four (vec4, float4 loads), at threadgroup sizes 32 to 1024
  occupancy  the scalar triad at a threadgroup of 256 with 1 KiB to 32 KiB of threadgroup memory
             declared and touched by every thread
  fma        8 independent chains of fused multiply-adds per thread, float32 and float16, at
             threadgroup sizes 32 to 1024

Every configuration is validated before it is timed: the triads against numpy on the first 1 Mi
elements, the FMA kernel by a short instantiation (3 steps) checked element by element against the
closed form, and the timed instantiation's every output against its fixed point. Thread 0 of every
kernel writes the threadgroup size it ran with and the SIMD-group width, which the run records.

Timing reuses Day 3's `time_call`: queue enough calls to last at least 50 ms, wait once, keep the
per-call time of each of 8 samples.

usage: python scripts/day04_worker.py --phase stream|occupancy|fma [--smoke]
--smoke runs every configuration on tiny inputs, validates it, and prints no timing at all. It
exists so the code can be checked to work without anyone seeing a number before the gated runs.
"""

from __future__ import annotations

import argparse
import json
import sys

import mlx.core as mx
import numpy as np
from day03_worker import close, entry, time_call

from inferlab import metal_sweep as ms

N_STREAM = 64 * 1024 * 1024
N_SMOKE = 64 * 1024
CHECK_HEAD = 1 << 20
SCALAR = 3.0
FMA_THREADS = 1 << 20
FMA_ITERATIONS = 2048
FMA_CHECK_ITERATIONS = 3
FMA_CHAINS = 8
FMA_X, FMA_Y = 0.75, 0.25  # each chain converges to y / (1 - x) = 1
FMA_DTYPES = {"fp32": (mx.float32, np.float32), "fp16": (mx.float16, np.float16)}

META = """
    if (thread_position_in_grid.x == 0) {
        meta[0] = threads_per_threadgroup.x;
        meta[1] = threads_per_simdgroup;
    }
"""

TRIAD_SCALAR = (
    """
    uint i = thread_position_in_grid.x;
    out[i] = a[i] + s[0] * b[i];
"""
    + META
)

TRIAD_VEC4 = (
    """
    uint i = thread_position_in_grid.x;
    const device float4* a4 = reinterpret_cast<const device float4*>(a);
    const device float4* b4 = reinterpret_cast<const device float4*>(b);
    device float4* o4 = reinterpret_cast<device float4*>(out);
    o4[i] = a4[i] + s[0] * b4[i];
"""
    + META
)

# SCRATCH floats of threadgroup memory. Each thread writes its triad value at a stride that spans
# the whole array, waits for the group, and stores its neighbour's value (lane l ^ 1), so the
# array is really used and the result proves the exchange happened.
TRIAD_SCRATCH = (
    """
    threadgroup float scratch[SCRATCH];
    uint i = thread_position_in_grid.x;
    uint l = thread_position_in_threadgroup.x;
    uint stride = SCRATCH / threads_per_threadgroup.x;
    scratch[l * stride] = a[i] + s[0] * b[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[i] = scratch[(l ^ 1) * stride];
"""
    + META
)


def fma_source(chains: int = FMA_CHAINS) -> str:
    lines = [
        "    uint i = thread_position_in_grid.x;",
        "    T x = coef[0];",
        "    T y = coef[1];",
        "    T base = init[i & 1023];",
    ]
    lines += [f"    T c{j} = base + T({j}) * T(0.01);" for j in range(chains)]
    lines.append("    for (int it = 0; it < ITERS; ++it) {")
    lines += [f"        c{j} = metal::fma(c{j}, x, y);" for j in range(chains)]
    lines.append("    }")
    total = " + ".join(f"c{j}" for j in range(chains))
    lines.append(f"    out[i] = {total};")
    return "\n".join(lines) + "\n" + META


def kernel(name: str, inputs: list[str], source: str):
    return mx.fast.metal_kernel(
        name=name, input_names=inputs, output_names=["out", "meta"], source=source
    )


def launch(k, inputs, template, grid: int, threadgroup: int, shape, dtype):
    return k(
        inputs=inputs,
        template=template,
        grid=(grid, 1, 1),
        threadgroup=(threadgroup, 1, 1),
        output_shapes=[shape, (2,)],
        output_dtypes=[dtype, mx.uint32],
    )


def meta_of(outputs) -> dict:
    meta = np.array(outputs[1])
    return {"dispatched_threadgroup": int(meta[0]), "simd_width": int(meta[1])}


def record(smoke: bool, flops, moved, fn, ok: bool, **extra) -> dict:
    outputs = fn()
    mx.eval(outputs)
    extra |= meta_of(outputs)
    if smoke:
        return {"validated": ok, **extra}
    return entry(flops, moved, time_call(fn), ok, **extra)


def triad_inputs(n: int):
    mx.random.seed(0)
    a = mx.random.uniform(shape=(n,))
    b = mx.random.uniform(shape=(n,))
    s = mx.array([SCALAR], dtype=mx.float32)
    mx.eval(a, b, s)
    head = min(n, CHECK_HEAD)
    want = np.array(a[:head]) + SCALAR * np.array(b[:head])
    return a, b, s, head, want


def phase_stream(smoke: bool) -> dict:
    n = N_SMOKE if smoke else N_STREAM
    a, b, s, head, want = triad_inputs(n)
    kernels = {
        "scalar": (kernel("day04_triad_scalar", ["a", "b", "s"], TRIAD_SCALAR), 1),
        "vec4": (kernel("day04_triad_vec4", ["a", "b", "s"], TRIAD_VEC4), 4),
    }
    configs = {}
    for variant, (k, per_thread) in kernels.items():
        for tg in ms.THREADGROUP_SIZES:

            def fn(k=k, tg=tg, per_thread=per_thread):
                return launch(k, [a, b, s], [], n // per_thread, tg, (n,), mx.float32)

            ok = close(np.array(fn()[0][:head]), want, 1e-6)
            configs[ms.stream_key(variant, tg)] = record(
                smoke, 2 * n, ms.stream_bytes(n), fn, ok,
                elements=n, variant=variant, threadgroup=tg,
            )  # fmt: skip
    return configs


def phase_occupancy(smoke: bool) -> dict:
    n = N_SMOKE if smoke else N_STREAM
    a, b, s, head, want = triad_inputs(n)
    swapped = want[np.arange(head) ^ 1]
    k = kernel("day04_triad_scratch", ["a", "b", "s"], TRIAD_SCRATCH)
    tg = ms.OCCUPANCY_THREADGROUP
    configs = {}
    for scratch_bytes in ms.SCRATCH_BYTES:
        template = [("SCRATCH", scratch_bytes // 4)]

        def fn(template=template):
            return launch(k, [a, b, s], template, n, tg, (n,), mx.float32)

        ok = close(np.array(fn()[0][:head]), swapped, 1e-6)
        configs[ms.occupancy_key(scratch_bytes)] = record(
            smoke, 2 * n, ms.stream_bytes(n), fn, ok,
            elements=n, scratch_bytes=scratch_bytes, threadgroup=tg,
        )  # fmt: skip
    return configs


def fma_expected(init: np.ndarray, iterations: int, np_dtype) -> np.ndarray:
    """Sum over chains of the closed form after `iterations` steps, from the rounded inputs."""
    base = init.astype(np_dtype).astype(np.float64)
    decay = FMA_X**iterations
    total = np.zeros_like(base)
    for j in range(FMA_CHAINS):
        start = (base + np_dtype(j) * np_dtype(0.01)).astype(np_dtype).astype(np.float64)
        total += 1.0 + (start - 1.0) * decay
    return total


def phase_fma(smoke: bool) -> dict:
    threads = N_SMOKE if smoke else FMA_THREADS
    iterations = 16 if smoke else FMA_ITERATIONS
    rng = np.random.default_rng(0)
    init_host = rng.uniform(0.0, 1.0, 1024)
    k = kernel("day04_fma", ["coef", "init"], fma_source())
    index = np.arange(threads) & 1023
    configs = {}
    for dtype_name, (mx_dtype, np_dtype) in FMA_DTYPES.items():
        coef = mx.array([FMA_X, FMA_Y], dtype=mx_dtype)
        init = mx.array(init_host.astype(np_dtype))
        mx.eval(coef, init)
        tolerance = 1e-5 if dtype_name == "fp32" else 5e-3
        width = 4 if dtype_name == "fp32" else 2  # bytes written per thread
        short = fma_expected(init_host, FMA_CHECK_ITERATIONS, np_dtype)[index]
        final = fma_expected(init_host, iterations, np_dtype)[index]
        for tg in ms.THREADGROUP_SIZES:

            def fn(tg=tg, steps=iterations, coef=coef, init=init, mx_dtype=mx_dtype):
                template = [("T", mx_dtype), ("ITERS", steps)]
                return launch(k, [coef, init], template, threads, tg, (threads,), mx_dtype)

            checked = np.array(fn(steps=FMA_CHECK_ITERATIONS)[0].astype(mx.float32))
            timed = np.array(fn()[0].astype(mx.float32))
            ok = bool(
                np.allclose(checked, short, rtol=tolerance, atol=0)
                and np.allclose(timed, final, rtol=tolerance, atol=0)
            )
            configs[ms.fma_key(dtype_name, tg)] = record(
                smoke, ms.fma_flops(threads, FMA_CHAINS, iterations), threads * width, fn, ok,
                threads=threads, chains=FMA_CHAINS, iterations=iterations,
                dtype=dtype_name, threadgroup=tg,
            )  # fmt: skip
    return configs


PHASES = {"stream": phase_stream, "occupancy": phase_occupancy, "fma": phase_fma}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    configs = PHASES[args.phase](args.smoke)
    print(
        json.dumps(
            {
                "phase": args.phase,
                "smoke": args.smoke,
                "mlx_version": getattr(mx, "__version__", None),
                "device": str(mx.default_device()),
                "configs": configs,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
