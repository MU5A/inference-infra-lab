"""Day 4's tables: numbers only from citable configurations, verdicts only from citable phases."""

from __future__ import annotations

from inferlab import metal_report as mr
from inferlab import metal_sweep as ms


def stat(median: float, low: float | None = None, high: float | None = None, cv=0.01) -> dict:
    return {
        "median": median,
        "min": median if low is None else low,
        "max": median if high is None else high,
        "cv": cv,
    }


def entry(rate: float, field: str, **flags) -> dict:
    other = "flops_per_s" if field == "bytes_per_s" else "bytes_per_s"
    base = {
        field: stat(rate, rate * 0.98, rate * 1.01),
        other: stat(1.0),
        "stable": True,
        "validated": True,
        "dispatched_as_asked": True,
        "simd_widths": [32],
    }
    return base | flags


def stream_result(overrides: dict | None = None) -> dict:
    s = {
        ms.stream_key(v, tg): entry(80e9 if v == "scalar" else 85e9, "bytes_per_s")
        for v in ms.STREAM_VARIANTS
        for tg in ms.THREADGROUP_SIZES
    }
    s.update(overrides or {})
    return {"summary": s}


def occupancy_result(rates=(80e9, 78e9, 70e9, 60e9, 55e9, 50e9), **flags) -> dict:
    return {
        "summary": {
            ms.occupancy_key(b): entry(r, "bytes_per_s", **flags)
            for b, r in zip(ms.SCRATCH_BYTES, rates, strict=True)
        }
    }


def fma_result() -> dict:
    return {
        "summary": {
            ms.fma_key(d, tg): entry(3.2e12 if d == "fp32" else 3.3e12, "flops_per_s")
            for d in ms.FMA_DTYPES
            for tg in ms.THREADGROUP_SIZES
        }
    }


MLX_TRIAD, MLX_FP32 = 87.2e9, 2.78e12


def test_citable_needs_all_three_flags():
    assert mr.citable(entry(1.0, "bytes_per_s"))
    for flag in ("stable", "validated", "dispatched_as_asked"):
        assert not mr.citable(entry(1.0, "bytes_per_s", **{flag: False}))
    assert not mr.citable(None)


def test_stream_table_shows_median_range_and_share_of_mlx():
    table = mr.stream_table(stream_result(), MLX_TRIAD)
    assert "| 256 | 80.0 (78.4 to 80.8) | 92% | 85.0 (83.3 to 85.8) | 97% |" in table
    assert len(table.splitlines()) == 2 + len(ms.THREADGROUP_SIZES)


def test_a_non_citable_configuration_shows_no_number():
    bad = {ms.stream_key("vec4", 32): entry(999e9, "bytes_per_s", stable=False)}
    table = mr.stream_table(stream_result(bad), MLX_TRIAD)
    row = next(line for line in table.splitlines() if line.startswith("| 32 |"))
    assert "999" not in row
    assert row.count(mr.NOT_CITABLE) == 2


def test_occupancy_table_is_relative_to_the_smallest_allocation():
    table = mr.occupancy_table(occupancy_result())
    assert "| 1 KiB | 80.0 (78.4 to 80.8) | 1.0% | 1.00 |" in table
    assert "| 32 KiB | 50.0 (49.0 to 50.5) | 1.0% | 0.62 |" in table


def test_fma_table_in_tflops():
    table = mr.fma_table(fma_result())
    assert "| 1024 | 3.20 (3.14 to 3.23) | 1.0% | 3.30 (3.23 to 3.33) | 1.0% |" in table


def test_dispatch_table_counts_what_ran_as_asked():
    occupancy = occupancy_result()
    occupancy["summary"][ms.occupancy_key(4096)]["dispatched_as_asked"] = False
    occupancy["summary"][ms.occupancy_key(4096)]["simd_widths"] = [16, 32]
    table = mr.dispatch_table({"occupancy": occupancy})
    assert "| occupancy | 6 | 5 of 6 | 16, 32 |" in table


def test_judge_evaluates_citable_phases():
    verdicts = mr.judge(stream_result(), occupancy_result(), fma_result(), MLX_TRIAD, MLX_FP32)
    assert [v["held"] for v in verdicts] == [True, True, True]


def test_judge_reports_a_failed_prediction():
    flat = occupancy_result(rates=(80e9,) * 6)
    verdicts = mr.judge(stream_result(), flat, fma_result(), MLX_TRIAD, MLX_FP32)
    assert verdicts[1]["held"] is False and verdicts[1]["failures"]
    table = mr.predictions_table(verdicts)
    assert "did not hold: 32768 bytes" in table


def test_judge_never_passes_a_phase_with_a_non_citable_configuration():
    bad = stream_result({ms.stream_key("scalar", 512): entry(80e9, "bytes_per_s", stable=False)})
    verdicts = mr.judge(bad, occupancy_result(), fma_result(), MLX_TRIAD, MLX_FP32)
    assert verdicts[0]["held"] is None
    assert "not evaluated" in mr.predictions_table(verdicts).splitlines()[2]
