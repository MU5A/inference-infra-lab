"""Day 4: threadgroup-size and occupancy sweeps of custom Metal kernels, the predictions (written
before any measurement) and the checks that judge them.

Three kernels, each written by hand in Metal and run through `mlx.core.fast.metal_kernel`:

- stream: the STREAM triad, out = a + s * b, on 64 Mi float32 elements (256 MiB per array), one
  element per thread (`scalar`) or four per thread through float4 loads (`vec4`), at threadgroup
  sizes 32 to 1024.
- occupancy: the scalar triad at a threadgroup of 256, with a threadgroup-memory array declared at
  1 KiB to 32 KiB and touched by every thread. More threadgroup memory per group means fewer
  groups can be resident on a GPU core at once (lower occupancy), and fewer memory loads in flight.
- fma: a compute-bound loop of fused multiply-adds, 8 independent chains per thread, in float32
  and float16, at threadgroup sizes 32 to 1024. It moves almost no memory, so it measures the
  arithmetic rate a hand-written kernel reaches.

Every kernel's thread 0 also writes the threadgroup size it was dispatched with and the SIMD-group
width (`threads_per_simdgroup`), so the sweep proves which size actually ran.

Predictions, written BEFORE any timing of these kernels on this machine. The inputs were Day 3's
MLX results (triad 87.2 GB/s, fp32 matrix multiply 2.78 TFLOPS) and my general understanding of
GPUs, not any run of this code. The ranges are deliberately wide.

1. The best hand-written triad (either variant, any threadgroup size) reaches at least 70% of
   Day 3's MLX triad median.
2. For each triad variant, every threadgroup size from 64 to 1024 is within 10% of that variant's
   best: a streaming kernel needs enough loads in flight, not a particular group size.
3. Declaring 32 KiB of threadgroup memory per group of 256 threads costs at least 20% of the
   bandwidth reached with 1 KiB, and no step up in memory makes it more than 10% faster.
4. The best float32 FMA rate is between 2.5 and 4.5 TFLOPS, and at least 0.95 times Day 3's MLX
   fp32 matrix-multiply roof (a pure FMA loop should not be slower than a full matrix multiply).
5. The best float16 FMA rate is between 0.8 and 1.25 times the best float32 rate: no doubling.
   If it holds it is consistent with Day 3's small half-precision gain, not proof of its cause.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .roofline import summarize_config

THREADGROUP_SIZES = (32, 64, 128, 256, 512, 1024)
STREAM_VARIANTS = ("scalar", "vec4")
OCCUPANCY_THREADGROUP = 256
SCRATCH_BYTES = (1024, 4096, 8192, 16384, 24576, 32768)
FMA_DTYPES = ("fp32", "fp16")

# Prediction thresholds, fixed before the first measurement.
STREAM_BEST_MIN_FRACTION_OF_MLX = 0.70
STREAM_FLAT_TOLERANCE = 0.10
STREAM_FLAT_FROM = 64
OCCUPANCY_MIN_DROP = 0.20
OCCUPANCY_MAX_RISE = 0.10
FP32_FMA_RANGE_TFLOPS = (2.5, 4.5)
FP32_FMA_MIN_FRACTION_OF_MATMUL = 0.95
FP16_VS_FP32_FMA_RANGE = (0.8, 1.25)


def stream_key(variant: str, threadgroup: int) -> str:
    return f"{variant}-tg{threadgroup}"


def occupancy_key(scratch_bytes: int) -> str:
    return f"scratch-{scratch_bytes}"


def fma_key(dtype: str, threadgroup: int) -> str:
    return f"{dtype}-tg{threadgroup}"


def stream_bytes(elements: int) -> int:
    """Least traffic of the triad: read a and b, write out, float32 each."""
    return 12 * elements


def fma_flops(threads: int, chains: int, iterations: int) -> float:
    """Each chain step is one fused multiply-add, counted as two operations. The final sum of the
    chains (chains - 1 adds per thread) is not counted, which understates the rate by under 0.1%."""
    return 2.0 * threads * chains * iterations


def summarize(runs: Sequence[Mapping], kind: str) -> dict:
    """Every configuration of one phase across independent process runs.

    `kind` is "bandwidth" or "compute"; it picks the stability limit, as on Day 3.
    """
    configs = [r["configs"] for r in runs]
    return {
        key: summarize_config(configs, key, kind) | _dispatch(configs, key) for key in configs[0]
    }


def _dispatch(configs: Sequence[Mapping], key: str) -> dict:
    """What every run says was dispatched. A size that did not run as requested is not valid."""
    asked = configs[0][key].get("threadgroup")
    seen = {c[key].get("dispatched_threadgroup") for c in configs}
    widths = {c[key].get("simd_width") for c in configs}
    return {
        "threadgroup": asked,
        "dispatched_as_asked": seen == {asked},
        "simd_widths": sorted(w for w in widths if w is not None),
    }


def _median(summary: Mapping, key: str, field: str) -> float:
    return summary[key][field]["median"]


def check_stream(summary: Mapping, mlx_triad_bytes_per_s: float) -> list[str]:
    """Predictions 1 and 2."""
    failures = []
    best = max(_median(summary, k, "bytes_per_s") for k in summary)
    floor = STREAM_BEST_MIN_FRACTION_OF_MLX * mlx_triad_bytes_per_s
    if best < floor:
        failures.append(
            f"best hand-written triad {best / 1e9:.1f} GB/s is {best / mlx_triad_bytes_per_s:.0%} "
            f"of the MLX triad, below the predicted {STREAM_BEST_MIN_FRACTION_OF_MLX:.0%}"
        )
    for variant in STREAM_VARIANTS:
        keys = [stream_key(variant, tg) for tg in THREADGROUP_SIZES]
        rates = {k: _median(summary, k, "bytes_per_s") for k in keys if k in summary}
        if not rates:
            failures.append(f"{variant}: no results")
            continue
        top = max(rates.values())
        for tg in THREADGROUP_SIZES:
            key = stream_key(variant, tg)
            if tg < STREAM_FLAT_FROM or key not in rates:
                continue
            if rates[key] < (1 - STREAM_FLAT_TOLERANCE) * top:
                failures.append(
                    f"{variant} threadgroup {tg}: {rates[key] / top:.0%} of the variant's best, "
                    f"more than {STREAM_FLAT_TOLERANCE:.0%} below it"
                )
    return failures


def check_occupancy(summary: Mapping) -> list[str]:
    """Prediction 3."""
    failures = []
    rates = [_median(summary, occupancy_key(b), "bytes_per_s") for b in SCRATCH_BYTES]
    drop = 1 - rates[-1] / rates[0]
    if drop < OCCUPANCY_MIN_DROP:
        failures.append(
            f"{SCRATCH_BYTES[-1]} bytes of threadgroup memory cost {drop:.0%} of the bandwidth "
            f"at {SCRATCH_BYTES[0]} bytes, less than the predicted {OCCUPANCY_MIN_DROP:.0%}"
        )
    for (small, low), (large, high) in zip(
        zip(SCRATCH_BYTES, rates, strict=True),
        zip(SCRATCH_BYTES[1:], rates[1:], strict=True),
        strict=False,
    ):
        if high > (1 + OCCUPANCY_MAX_RISE) * low:
            failures.append(
                f"going from {small} to {large} bytes made it {high / low - 1:.0%} faster, "
                f"more than the {OCCUPANCY_MAX_RISE:.0%} allowed"
            )
    return failures


def best_fma(summary: Mapping, dtype: str) -> float:
    return max(_median(summary, fma_key(dtype, tg), "flops_per_s") for tg in THREADGROUP_SIZES)


def check_fma(summary: Mapping, mlx_fp32_matmul_flops: float) -> list[str]:
    """Predictions 4 and 5."""
    failures = []
    fp32, fp16 = best_fma(summary, "fp32"), best_fma(summary, "fp16")
    low, high = FP32_FMA_RANGE_TFLOPS
    if not low <= fp32 / 1e12 <= high:
        failures.append(
            f"best fp32 FMA {fp32 / 1e12:.2f} TFLOPS is outside the predicted {low:g} to {high:g}"
        )
    if fp32 < FP32_FMA_MIN_FRACTION_OF_MATMUL * mlx_fp32_matmul_flops:
        failures.append(
            f"best fp32 FMA is {fp32 / mlx_fp32_matmul_flops:.2f} times the MLX fp32 matrix "
            f"multiply, below the predicted {FP32_FMA_MIN_FRACTION_OF_MATMUL:g}"
        )
    low, high = FP16_VS_FP32_FMA_RANGE
    ratio = fp16 / fp32
    if not low <= ratio <= high:
        failures.append(
            f"fp16 FMA is {ratio:.2f} times fp32, outside the predicted {low:g} to {high:g}"
        )
    return failures


PREDICTIONS = (
    ("stream", "Best hand-written triad at least 70% of Day 3's MLX triad; threadgroup sizes "
     "64 to 1024 within 10% of each variant's best"),
    ("occupancy", "32 KiB of threadgroup memory costs at least 20% of the 1 KiB bandwidth; no "
     "step up in memory more than 10% faster"),
    ("fma", "Best fp32 FMA between 2.5 and 4.5 TFLOPS and at least 0.95 times Day 3's fp32 "
     "matrix multiply; fp16 between 0.8 and 1.25 times fp32"),
)  # fmt: skip


def verdicts(
    stream: Mapping, occupancy: Mapping, fma: Mapping, mlx_triad: float, mlx_fp32: float
) -> list[dict]:
    failures = {
        "stream": check_stream(stream, mlx_triad),
        "occupancy": check_occupancy(occupancy),
        "fma": check_fma(fma, mlx_fp32),
    }
    return [
        {
            "phase": phase,
            "prediction": text,
            "held": not failures[phase],
            "failures": failures[phase],
        }
        for phase, text in PREDICTIONS
    ]
