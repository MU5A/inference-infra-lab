"""Analysis for the Day 2 memory-hierarchy measurements. Pure functions, no I/O.

A pointer-chasing sweep gives latency against working-set size. Each level of the memory
hierarchy shows up as a plateau, and a capacity as the point where the curve steps up to the next
one. These functions find the plateaus in the data, so the levels are read off the measurements
instead of being assumed from a datasheet.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .stats import coefficient_of_variation, percentile

# Within a plateau, latencies stay within this fraction of the plateau's median.
PLATEAU_TOLERANCE = 0.30
MIN_PLATEAU_POINTS = 3

# How repeatable each kind of measurement has to be across independent runs. Chosen before the
# first run, not fitted to its results. Cold reads depend on the SSD, which is noisier.
MAX_CV = {"bandwidth": 0.10, "latency": 0.10, "compute": 0.10, "cold_read": 0.25}


@dataclass(frozen=True)
class Plateau:
    start_bytes: int
    end_bytes: int
    median_ns: float
    points: int


def plateaus(
    curve: Sequence[tuple[int, float]],
    tolerance: float = PLATEAU_TOLERANCE,
    min_points: int = MIN_PLATEAU_POINTS,
) -> list[Plateau]:
    """Group a (bytes, ns) curve into plateaus of similar latency.

    A plateau is a run of at least `min_points` consecutive points that all lie within
    `tolerance` of the run's median. Points on the slope between two plateaus belong to no
    plateau. `end_bytes` is the largest size still on the plateau, a conservative estimate of the
    capacity of that level, because the curve rises a little after the true capacity is passed.
    """
    ordered = sorted(curve)
    found: list[Plateau] = []
    run: list[tuple[int, float]] = []

    def close() -> None:
        if len(run) >= min_points:
            found.append(
                Plateau(run[0][0], run[-1][0], percentile([ns for _, ns in run], 50), len(run))
            )

    for size, ns in ordered:
        if run:
            centre = percentile([v for _, v in run], 50)
            if abs(ns - centre) <= tolerance * centre:
                run.append((size, ns))
                continue
            close()
            run = []
        run.append((size, ns))
    close()
    return found


def merge_latency_runs(runs: Sequence[Sequence[dict]]) -> list[dict]:
    """Combine several sweeps of the same sizes: median latency per size and its variation."""
    by_size: dict[int, list[float]] = {}
    for run in runs:
        for point in run:
            by_size.setdefault(point["bytes"], []).append(point["ns_median"])
    merged = []
    for size in sorted(by_size):
        values = by_size[size]
        merged.append(
            {
                "bytes": size,
                "ns": percentile(values, 50),
                "min": min(values),
                "max": max(values),
                "cv": coefficient_of_variation(values),
                "runs": len(values),
            }
        )
    return merged


def across_runs(values: Sequence[float], kind: str) -> dict:
    """Median, range and variation of one metric over independent runs, and whether it is stable."""
    cv = coefficient_of_variation(values)
    limit = MAX_CV[kind]
    return {
        "median": percentile(values, 50),
        "min": min(values),
        "max": max(values),
        "cv": cv,
        "runs": len(values),
        "stable": cv is None or cv <= limit,
        "cv_limit": limit,
    }


def latency_summary(sweeps: Sequence[dict]) -> dict:
    """Combine repeated pointer-chase sweeps into a curve, its plateaus, and a verdict on stability.

    The verdict requires every run to find the same number of plateaus and each plateau's latency
    to repeat within the limit. A set of sweeps that disagrees about the shape of the hierarchy is
    not a measurement of it.
    """
    per_run = [plateaus([(p["bytes"], p["ns_median"]) for p in s["points"]]) for s in sweeps]
    merged = merge_latency_runs([s["points"] for s in sweeps])
    consensus = plateaus([(m["bytes"], m["ns"]) for m in merged])
    same_structure = len({len(p) for p in per_run}) == 1
    stability = []
    if same_structure and per_run and per_run[0]:
        for index in range(len(per_run[0])):
            entry = across_runs([run[index].median_ns for run in per_run], "latency")
            entry["end_bytes_per_run"] = [run[index].end_bytes for run in per_run]
            stability.append(entry)
    return {
        "merged": merged,
        "plateaus": [asdict(p) for p in consensus],
        "plateaus_per_run": [len(p) for p in per_run],
        "same_structure": same_structure,
        "plateau_stability": stability,
        "stable": bool(same_structure and stability and all(e["stable"] for e in stability)),
    }


def check_cold_predictions(trials: Sequence[dict]) -> list[str]:
    """Predictions written BEFORE the redesigned cold-read experiment was run.

    The first cold run measured a memory-mapped touch at 0.07 GB/s, about 100 times slower than the
    same file once it was cached. That could mean the disk is slow or that page-by-page faulting
    is. So the redesigned run adds a cold sequential read() as a control, and two prefetch hints.
    Predictions: a cold sequential read is at least 5 times faster than a cold plain mapped touch
    (the touch is fault-latency bound, the read is bandwidth bound), a willneed hint is at least 3
    times faster than a plain touch, and every cold pass is at least 1.5 times slower than its warm
    pass. A violated prediction is a finding.
    """
    failures = []
    for index, trial in enumerate(trials, start=1):
        plain = trial["cold_touch"]["gbs"]
        if trial["cold_read"]["gbs"] < 5 * plain:
            failures.append(
                f"trial {index}: a cold sequential read was not 5x faster than a cold touch"
            )
        if trial["cold_willneed"]["gbs"] < 3 * plain:
            failures.append(f"trial {index}: a willneed hint did not make a cold touch 3x faster")
        for name in ("cold_touch", "cold_read", "cold_willneed", "cold_sequential"):
            if not trial["checks"][name]["really_cold"]:
                failures.append(f"trial {index}: {name} was not clearly colder than its warm pass")
    return failures


def cold_vs_warm(cold_gbs: float, warm_gbs: float, min_ratio: float = 1.5) -> dict:
    """Whether a first read really was cold. If the second pass is not clearly faster than the
    first, the file was already in the page cache and the 'cold' number is not a cold number."""
    ratio = warm_gbs / cold_gbs if cold_gbs > 0 else float("inf")
    return {"ratio": ratio, "really_cold": ratio >= min_ratio, "min_ratio": min_ratio}
