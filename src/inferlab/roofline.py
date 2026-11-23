"""The roofline model: a calculator, the Day 3 predictions (written before measuring), and checks.

A kernel that does F floating-point operations and moves B bytes has arithmetic intensity I = F / B.
On a machine with peak compute P (FLOP/s) and peak memory bandwidth W (bytes/s) it cannot run
faster than min(P, W * I). The ridge point P / W is the intensity where the two limits meet: below
it a kernel is limited by memory, above it by compute.

Predictions for the measurements on this chip's GPU, written BEFORE the first MLX run. They come
from Day 2 (CPU triad 80 to 86 GB/s, Accelerate 1.3 to 1.65 TFLOPS) and from memory of public MLX
numbers for this chip, not from any run here. The ranges are wide on purpose.

1. Streaming triad through MLX, 256 MiB arrays: between 60 and 100 GB/s.
2. Single-precision square matrix multiply, n = 4096: between 1.0 and 4.0 TFLOPS.
3. Half precision at n = 4096 is at least 0.9 times as fast as single precision.
4. Single precision at n = 4096 is at least 1.2 times as fast as at n = 512 (launch cost).
5. Decode-shaped products (batch B times a 8192 x 8192 half-precision weight matrix), with the
   measured roofs: no point above its roofline by more than 5%; for B at most a quarter of the
   predicted knee, at least 60% of the measured bandwidth roof; for B at least four times the
   knee, at least 50% of the measured half-precision compute roof.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .memory import across_runs

BANDWIDTH_RANGE_GBS = (60.0, 100.0)
FP32_4096_RANGE_TFLOPS = (1.0, 4.0)
FP16_VS_FP32_MIN_RATIO = 0.9
LARGE_VS_SMALL_MIN_RATIO = 1.2
ROOF_TOLERANCE = 0.05
MEMORY_REGION_KNEE_FRACTION = 0.25
MEMORY_REGION_MIN_BANDWIDTH_FRACTION = 0.60
COMPUTE_REGION_KNEE_MULTIPLE = 4.0
COMPUTE_REGION_MIN_PEAK_FRACTION = 0.50

UNIT_SCALE = {
    "GB/s": 1e9,
    "TB/s": 1e12,
    "GFLOPS": 1e9,
    "TFLOPS": 1e12,
    "PFLOPS": 1e15,
}
HOW = ("printed", "derived")


@dataclass(frozen=True)
class Roof:
    """One machine's two limits: peak compute in FLOP/s and peak bandwidth in bytes/s."""

    name: str
    peak_flops: float
    bandwidth: float

    def __post_init__(self) -> None:
        if self.peak_flops <= 0 or self.bandwidth <= 0:
            raise ValueError("a roof needs a positive compute peak and a positive bandwidth")

    @property
    def ridge(self) -> float:
        """Intensity in FLOP per byte where the memory limit meets the compute limit."""
        return self.peak_flops / self.bandwidth

    def attainable(self, intensity: float) -> float:
        return min(self.peak_flops, self.bandwidth * intensity)

    def bound(self, intensity: float) -> str:
        return "memory" if intensity < self.ridge else "compute"


def gemm_flops(m: int, n: int, k: int) -> float:
    """Floating-point operations of an (m x k) times (k x n) product: one multiply, one add each."""
    return 2.0 * m * n * k


def gemm_bytes(m: int, n: int, k: int, in_bytes: float, out_bytes: float | None = None) -> float:
    """Least memory traffic of that product: each of the three matrices touched exactly once."""
    out = in_bytes if out_bytes is None else out_bytes
    return (m * k + k * n) * in_bytes + m * n * out


def gemm_intensity(
    m: int, n: int, k: int, in_bytes: float, out_bytes: float | None = None
) -> float:
    return gemm_flops(m, n, k) / gemm_bytes(m, n, k, in_bytes, out_bytes)


def knee_batch(
    k: int, n: int, weight_bytes: float, act_bytes: float, out_bytes: float, ridge: float
) -> float | None:
    """Batch size B at which (B x k) times (k x n) reaches the ridge point, or None.

    Solves 2 B k n = ridge * (k n weight + B k act + B n out) for B. With the weights dominating
    the traffic this is ridge * weight_bytes / 2: a decode batch must be about that large before the
    GPU stops waiting for memory.
    """
    denominator = 2.0 * k * n - ridge * (k * act_bytes + n * out_bytes)
    if denominator <= 0:
        return None
    return ridge * k * n * weight_bytes / denominator


def place(roof: Roof, intensity: float, achieved_flops: float) -> dict:
    """Where a measured point sits against a roof."""
    attainable = roof.attainable(intensity)
    fraction = achieved_flops / attainable
    return {
        "intensity": intensity,
        "attainable_flops": attainable,
        "achieved_flops": achieved_flops,
        "fraction_of_roof": fraction,
        "bound": roof.bound(intensity),
        "above_roof": fraction > 1.0 + ROOF_TOLERANCE,
    }


def summarize_config(runs: Sequence[Mapping], key: str, kind: str) -> dict:
    """One measured configuration across independent process runs.

    Each run reports the work per call (`flops`, `bytes`) and the fastest per-call time among its
    samples (`best_s`). The rate is work over the fastest time, then the median over runs, with the
    variation judged against the limit for `kind` ("bandwidth" or "compute").
    """
    first = runs[0][key]
    flops, moved = first["flops"], first["bytes"]
    flop_rates = [r[key]["flops"] / r[key]["best_s"] for r in runs]
    byte_rates = [r[key]["bytes"] / r[key]["best_s"] for r in runs]
    stats = across_runs(byte_rates if kind == "bandwidth" else flop_rates, kind)
    return {
        "flops_per_call": flops,
        "bytes_per_call": moved,
        "intensity": flops / moved,
        "flops_per_s": across_runs(flop_rates, "compute"),
        "bytes_per_s": across_runs(byte_rates, "bandwidth"),
        "stable": stats["stable"],
        "validated": all(r[key].get("validated", True) for r in runs),
    }


def _in_range(value: float, low: float, high: float) -> bool:
    return low <= value <= high


def check_bandwidth(gbs: float) -> list[str]:
    low, high = BANDWIDTH_RANGE_GBS
    if _in_range(gbs, low, high):
        return []
    return [f"GPU triad bandwidth {gbs:.1f} GB/s is outside the predicted {low:g} to {high:g}"]


def check_matmul(fp32_large_tflops: float, fp16_large_tflops: float, fp32_small_tflops: float):
    failures = []
    low, high = FP32_4096_RANGE_TFLOPS
    if not _in_range(fp32_large_tflops, low, high):
        failures.append(
            f"fp32 n=4096 {fp32_large_tflops:.2f} TFLOPS is outside the predicted "
            f"{low:g} to {high:g}"
        )
    if fp16_large_tflops < FP16_VS_FP32_MIN_RATIO * fp32_large_tflops:
        failures.append("fp16 n=4096 was not at least 0.9 times as fast as fp32")
    if fp32_large_tflops < LARGE_VS_SMALL_MIN_RATIO * fp32_small_tflops:
        failures.append("fp32 n=4096 was not at least 1.2 times as fast as n=512")
    return failures


def check_skinny(points: Sequence[Mapping], roof: Roof, knee: float | None) -> list[str]:
    """Judge the decode-shaped sweep against the measured roofline.

    `points` carry `batch`, `intensity`, `flops_per_s` and `bytes_per_s`. `knee` is the batch size
    predicted to reach the ridge point; without one only the roofline ceiling is checked.
    """
    failures = []
    for p in points:
        placed = place(roof, p["intensity"], p["flops_per_s"])
        if placed["above_roof"]:
            failures.append(
                f"batch {p['batch']}: {placed['fraction_of_roof']:.0%} of its roofline, above it"
            )
        if knee is None:
            continue
        if p["batch"] <= MEMORY_REGION_KNEE_FRACTION * knee:
            floor = MEMORY_REGION_MIN_BANDWIDTH_FRACTION * roof.bandwidth
            if p["bytes_per_s"] < floor:
                failures.append(
                    f"batch {p['batch']}: memory-bound but reached only "
                    f"{p['bytes_per_s'] / roof.bandwidth:.0%} of the bandwidth roof"
                )
        if p["batch"] >= COMPUTE_REGION_KNEE_MULTIPLE * knee:
            floor = COMPUTE_REGION_MIN_PEAK_FRACTION * roof.peak_flops
            if p["flops_per_s"] < floor:
                failures.append(
                    f"batch {p['batch']}: compute-bound but reached only "
                    f"{p['flops_per_s'] / roof.peak_flops:.0%} of the compute roof"
                )
    return failures


def to_si(value: float, unit: str) -> float:
    """A printed figure in the unit the vendor used, as bytes/s or FLOP/s."""
    return value * UNIT_SCALE[unit]


def validate_specs(data: Mapping) -> list[str]:
    """Every specification figure must say where it came from, when, and whether it was computed."""
    problems: list[str] = []
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(data.get("fetched", ""))):
        problems.append("the file has no fetch date")
    if data.get("label") != "L":
        problems.append("specification figures must be labelled L")
    for device in data.get("devices", []):
        name = device.get("name", "?")
        figures = [("memory_bandwidth", device.get("memory_bandwidth"))]
        figures += [(f"compute.{k}", v) for k, v in (device.get("compute") or {}).items()]
        for label, fig in figures:
            where = f"{name} {label}"
            if not fig:
                problems.append(f"{where}: missing")
                continue
            if fig.get("unit") not in UNIT_SCALE:
                problems.append(f"{where}: unknown unit {fig.get('unit')!r}")
            if not isinstance(fig.get("value"), int | float) or fig["value"] <= 0:
                problems.append(f"{where}: value is not a positive number")
            if not str(fig.get("source", "")).startswith("https://"):
                problems.append(f"{where}: no source URL")
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(fig.get("fetched", ""))):
                problems.append(f"{where}: no fetch date")
            if fig.get("how") not in HOW:
                problems.append(f"{where}: must say whether it is printed or derived")
            if fig.get("how") == "printed" and not fig.get("as_printed"):
                problems.append(f"{where}: a printed figure must give the text as printed")
            if fig.get("how") == "derived" and not (fig.get("from") and fig.get("rule")):
                problems.append(
                    f"{where}: a derived figure must give what it came from and the rule"
                )
    return problems


def load_specs(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    problems = validate_specs(data)
    if problems:
        raise ValueError("invalid specification file: " + "; ".join(problems))
    return data


def spec_roof(device: Mapping, precision: str) -> Roof | None:
    """The roof for one device and compute entry, or None if the vendor printed no such figure."""
    entry = (device.get("compute") or {}).get(precision)
    if not entry:
        return None
    bw = device["memory_bandwidth"]
    return Roof(
        f"{device['name']} ({precision})",
        to_si(entry["value"], entry["unit"]),
        to_si(bw["value"], bw["unit"]),
    )
