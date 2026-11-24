"""Day 4's predictions and checks, on synthetic results with known answers."""

from __future__ import annotations

import pytest

from inferlab import metal_sweep as ms


def run(configs: dict) -> dict:
    return {"configs": configs}


def entry(rate: float, kind: str, tg: int, dispatched: int | None = None, width: int = 32) -> dict:
    """One configuration whose rate is exactly `rate` (bytes/s or FLOP/s) over a 1 s call."""
    work = {"flops": rate, "bytes": 1.0} if kind == "compute" else {"flops": 1.0, "bytes": rate}
    return work | {
        "best_s": 1.0,
        "validated": True,
        "threadgroup": tg,
        "dispatched_threadgroup": tg if dispatched is None else dispatched,
        "simd_width": width,
    }


def stream_runs(rates: dict[str, float], copies: int = 3) -> list[dict]:
    configs = {}
    for variant in ms.STREAM_VARIANTS:
        for tg in ms.THREADGROUP_SIZES:
            key = ms.stream_key(variant, tg)
            configs[key] = entry(rates.get(key, 80e9), "bandwidth", tg)
    return [run(configs) for _ in range(copies)]


def occupancy_runs(rates: list[float]) -> list[dict]:
    configs = {
        ms.occupancy_key(b): entry(r, "bandwidth", ms.OCCUPANCY_THREADGROUP)
        for b, r in zip(ms.SCRATCH_BYTES, rates, strict=True)
    }
    return [run(configs) for _ in range(3)]


def fma_runs(fp32: float, fp16: float) -> list[dict]:
    configs = {}
    for tg in ms.THREADGROUP_SIZES:
        configs[ms.fma_key("fp32", tg)] = entry(fp32 if tg == 256 else fp32 / 2, "compute", tg)
        configs[ms.fma_key("fp16", tg)] = entry(fp16 if tg == 512 else fp16 / 2, "compute", tg)
    return [run(configs) for _ in range(3)]


MLX_TRIAD = 87.2e9
MLX_FP32 = 2.78e12


def test_work_formulas():
    assert ms.stream_bytes(10) == 120
    assert ms.fma_flops(4, 8, 16) == 2 * 4 * 8 * 16


def test_keys_are_distinct_and_readable():
    assert ms.stream_key("vec4", 256) == "vec4-tg256"
    assert ms.occupancy_key(4096) == "scratch-4096"
    assert ms.fma_key("fp16", 32) == "fp16-tg32"


def test_summary_takes_the_median_over_runs_and_records_dispatch():
    runs = [run({"k": entry(r, "bandwidth", 64)}) for r in (80e9, 90e9, 85e9)]
    s = ms.summarize(runs, "bandwidth")["k"]
    assert s["bytes_per_s"]["median"] == pytest.approx(85e9)
    assert s["dispatched_as_asked"] is True
    assert s["simd_widths"] == [32]
    assert s["stable"] is True


def test_a_threadgroup_that_ran_at_another_size_is_flagged():
    runs = [run({"k": entry(80e9, "bandwidth", 1024, dispatched=d)}) for d in (1024, 1024, 512)]
    assert ms.summarize(runs, "bandwidth")["k"]["dispatched_as_asked"] is False


def test_an_unstable_configuration_is_marked():
    runs = [run({"k": entry(r, "bandwidth", 64)}) for r in (60e9, 90e9, 85e9)]
    assert ms.summarize(runs, "bandwidth")["k"]["stable"] is False


def test_stream_holds_when_flat_and_fast():
    summary = ms.summarize(stream_runs({}), "bandwidth")
    assert ms.check_stream(summary, MLX_TRIAD) == []


def test_stream_fails_when_too_slow_against_mlx():
    rates = {ms.stream_key(v, tg): 50e9 for v in ms.STREAM_VARIANTS for tg in ms.THREADGROUP_SIZES}
    failures = ms.check_stream(ms.summarize(stream_runs(rates), "bandwidth"), MLX_TRIAD)
    assert any("below the predicted 70%" in f for f in failures)


def test_stream_just_above_the_mlx_floor_holds():
    rates = {
        ms.stream_key(v, tg): 0.701 * MLX_TRIAD
        for v in ms.STREAM_VARIANTS
        for tg in ms.THREADGROUP_SIZES
    }
    assert ms.check_stream(ms.summarize(stream_runs(rates), "bandwidth"), MLX_TRIAD) == []


def test_stream_fails_when_one_size_from_64_up_is_far_below_the_best():
    failures = ms.check_stream(
        ms.summarize(stream_runs({ms.stream_key("vec4", 1024): 70e9}), "bandwidth"), MLX_TRIAD
    )
    assert failures == [failures[0]] and "vec4 threadgroup 1024" in failures[0]


def test_stream_threadgroup_32_is_not_judged_for_flatness():
    summary = ms.summarize(stream_runs({ms.stream_key("scalar", 32): 40e9}), "bandwidth")
    assert ms.check_stream(summary, MLX_TRIAD) == []


def test_stream_just_inside_ten_percent_below_holds():
    summary = ms.summarize(stream_runs({ms.stream_key("scalar", 64): 72.01e9}), "bandwidth")
    assert ms.check_stream(summary, MLX_TRIAD) == []


def test_occupancy_holds_on_a_falling_curve():
    rates = [80e9, 79e9, 75e9, 70e9, 64e9, 60e9]
    assert ms.check_occupancy(ms.summarize(occupancy_runs(rates), "bandwidth")) == []


def test_occupancy_fails_on_a_flat_curve():
    failures = ms.check_occupancy(ms.summarize(occupancy_runs([80e9] * 6), "bandwidth"))
    assert len(failures) == 1 and "less than the predicted 20%" in failures[0]


def test_occupancy_just_over_twenty_percent_drop_holds():
    rates = [80e9, 80e9, 80e9, 80e9, 80e9, 63.99e9]
    assert ms.check_occupancy(ms.summarize(occupancy_runs(rates), "bandwidth")) == []


def test_occupancy_fails_when_more_memory_is_much_faster():
    rates = [80e9, 60e9, 70e9, 60e9, 60e9, 50e9]
    failures = ms.check_occupancy(ms.summarize(occupancy_runs(rates), "bandwidth"))
    assert failures == [
        "going from 4096 to 8192 bytes made it 17% faster, more than the 10% allowed"
    ]


def test_fma_holds_in_range():
    assert ms.check_fma(ms.summarize(fma_runs(3.0e12, 3.1e12), "compute"), MLX_FP32) == []


def test_fma_best_is_the_best_threadgroup():
    summary = ms.summarize(fma_runs(3.0e12, 3.1e12), "compute")
    assert ms.best_fma(summary, "fp32") == pytest.approx(3.0e12)
    assert ms.best_fma(summary, "fp16") == pytest.approx(3.1e12)


@pytest.mark.parametrize(
    ("fp32", "fp16", "words"),
    [
        (5.0e12, 5.0e12, "outside the predicted 2.5 to 4.5"),
        (2.0e12, 2.0e12, "outside the predicted 2.5 to 4.5"),
        (2.6e12, 2.6e12, "below the predicted 0.95"),
        (3.0e12, 6.0e12, "fp16 FMA is 2.00 times fp32"),
        (3.0e12, 2.0e12, "fp16 FMA is 0.67 times fp32"),
    ],
)
def test_fma_failures(fp32, fp16, words):
    failures = ms.check_fma(ms.summarize(fma_runs(fp32, fp16), "compute"), MLX_FP32)
    assert any(words in f for f in failures), failures


def test_fma_edges_hold():
    assert ms.check_fma(ms.summarize(fma_runs(2.5e12, 3.125e12), "compute"), 2.5e12) == []
    assert ms.check_fma(ms.summarize(fma_runs(4.0e12, 3.2e12), "compute"), MLX_FP32) == []


def test_verdicts_name_every_phase_and_carry_the_failures():
    v = ms.verdicts(
        ms.summarize(stream_runs({}), "bandwidth"),
        ms.summarize(occupancy_runs([80e9] * 6), "bandwidth"),
        ms.summarize(fma_runs(3.0e12, 3.1e12), "compute"),
        MLX_TRIAD,
        MLX_FP32,
    )
    assert [x["phase"] for x in v] == ["stream", "occupancy", "fma"]
    assert [x["held"] for x in v] == [True, False, True]
    assert v[1]["failures"]
