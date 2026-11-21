"""Small, dependency-free statistics for latency samples.

Percentiles use linear interpolation between the two nearest ranks, which is the same definition
NumPy uses by default, so results can be cross-checked against it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile (0 to 100) of `values`, by linear interpolation. Raises on no data."""
    if not values:
        raise ValueError("percentile of an empty sample")
    if not 0 <= q <= 100:
        raise ValueError(f"q must be between 0 and 100, got {q}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q / 100
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    """Count, median, p95, p99, min, max and mean. An empty sample gives None for each statistic."""
    data = [float(v) for v in values]
    if not data:
        return {
            "n": 0,
            "median": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
            "mean": None,
        }
    return {
        "n": len(data),
        "median": percentile(data, 50),
        "p95": percentile(data, 95),
        "p99": percentile(data, 99),
        "min": min(data),
        "max": max(data),
        "mean": sum(data) / len(data),
    }
