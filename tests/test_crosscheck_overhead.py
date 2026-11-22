"""The predictions for Day 1's follow-ups, and the overhead measurement itself."""

from __future__ import annotations

import pytest

from inferlab import crosscheck as cc
from inferlab import overhead


def block(ttft_ms, prompt=100, cached=0, requests=5, failed=0, stable=True):
    return {
        "summary": {
            "prompt_tokens": [prompt],
            "ttft_s": {"median": ttft_ms / 1e3},
            "cached_tokens_seen": cached * (requests - failed),
            "requests": requests,
            "failed": failed,
            "stable": stable,
        }
    }


def reports(cold_ms=100.0, warm_ms=10.0, prompt=100, warm_cached=None, cold_cached=0, **warm_kw):
    warm_cached = prompt - 1 if warm_cached is None else warm_cached
    return (
        {"results": {"p": block(cold_ms, prompt, cold_cached)}},
        {"results": {"p": block(warm_ms, prompt, warm_cached, **warm_kw)}},
    )


class TestCache:
    def test_the_predicted_outcome_has_no_failures(self):
        comparison = cc.compare_cache(*reports())
        assert cc.check_cache(comparison) == []
        row = comparison["p"]
        assert row["speedup"] == pytest.approx(10.0)
        assert row["warm_cached_per_request"] == 99

    def test_a_cold_run_that_reused_tokens_was_not_cold(self):
        failures = cc.check_cache(cc.compare_cache(*reports(cold_cached=40)))
        assert any("was not cold" in f for f in failures)

    def test_a_small_speedup_is_a_finding(self):
        assert any(
            "not 5x lower" in f for f in cc.check_cache(cc.compare_cache(*reports(warm_ms=30)))
        )

    def test_the_speedup_boundary_is_inclusive(self):
        assert cc.check_cache(cc.compare_cache(*reports(cold_ms=100, warm_ms=20))) == []

    @pytest.mark.parametrize("cached", [0, 50, 100])
    def test_reusing_anything_other_than_all_but_one_token_is_a_finding(self, cached):
        failures = cc.check_cache(cc.compare_cache(*reports(warm_cached=cached)))
        assert any("all but one" in f for f in failures)

    def test_failed_requests_do_not_dilute_the_cached_token_average(self):
        comparison = cc.compare_cache(*reports(warm_cached=99, failed=1))
        assert comparison["p"]["warm_cached_per_request"] == 99

    def test_unstable_runs_cannot_be_cited(self):
        failures = cc.check_cache(cc.compare_cache(*reports(stable=False)))
        assert any("not repeatable" in f for f in failures)


def bench_rows(pp=3200.0, tg=160.0, pp_sd=40.0, tg_sd=3.0):
    return [
        {"n_prompt": 512, "n_gen": 0, "avg_ts": pp, "stddev_ts": pp_sd, "samples_ts": [pp] * 3},
        {"n_prompt": 0, "n_gen": 128, "avg_ts": tg, "stddev_ts": tg_sd, "samples_ts": [tg] * 3},
    ]


class TestLlamaBench:
    def test_rows_are_keyed_by_what_they_measure(self):
        summary = cc.summarize_llama_bench(bench_rows())
        assert set(summary) == {"pp512", "tg128"}
        assert summary["tg128"]["cv"] == pytest.approx(3 / 160)

    def test_a_run_inside_the_predictions_has_no_failures(self):
        summary = cc.summarize_llama_bench(bench_rows())
        assert (
            cc.check_llama_bench(summary, harness_decode_tps=152.0, server_prefill_tps=3005.0) == []
        )

    def test_the_real_failure_is_refused_on_variation_alone(self):
        """The disturbed run: decode averaged 60.5 tokens/s with a deviation of 50.2."""
        summary = cc.summarize_llama_bench(bench_rows(tg=60.5, tg_sd=50.2))
        failures = cc.check_llama_bench(summary, 152.0, 3005.0)
        assert any("tg128: variation 83.0% is above the limit" in f for f in failures)

    def test_decode_far_from_the_harness_is_a_finding(self):
        summary = cc.summarize_llama_bench(bench_rows(tg=200.0))
        assert any("decode" in f for f in cc.check_llama_bench(summary, 152.0, 3005.0))

    def test_prefill_far_from_the_server_rate_is_a_finding(self):
        summary = cc.summarize_llama_bench(bench_rows(pp=4000.0))
        assert any("prompt processing" in f for f in cc.check_llama_bench(summary, 152.0, 3005.0))

    def test_the_tolerances_are_inclusive(self):
        # 110 against 100 is exactly 10%, and 2300 against 2000 is exactly 15%
        summary = cc.summarize_llama_bench(bench_rows(tg=110.0, pp=2300.0))
        assert cc.check_llama_bench(summary, 100.0, 2000.0) == []

    def test_limits_were_fixed_before_the_data(self):
        assert (
            cc.CACHE_MIN_SPEEDUP,
            cc.DECODE_TOLERANCE,
            cc.PREFILL_TOLERANCE,
            cc.MAX_BENCH_CV,
        ) == (
            5.0,
            0.10,
            0.15,
            0.05,
        )


class TestReference:
    def test_a_probe_near_the_baseline_passes_and_a_faster_one_does_too(self):
        assert cc.check_reference(150.0, 152.5) == []
        assert cc.check_reference(190.0, 152.5) == []

    def test_the_floor_is_inclusive(self):
        assert cc.check_reference(120.0, 150.0) == []  # exactly 20% below
        assert cc.check_reference(119.9, 150.0) != []

    def test_the_real_third_attempt_is_refused_even_though_its_repeats_were_steady(self):
        (problem,) = cc.check_reference(55.65, 152.55)
        assert "56 tok/s" in problem and "below 122" in problem

    def test_no_measurement_is_refused(self):
        assert cc.check_reference(None, 150.0) == ["no reference decode speed was measured"]

    def test_the_tolerance_is_the_one_the_log_records(self):
        assert cc.REFERENCE_TOLERANCE == 0.20


class TestOverhead:
    def runs(self, *medians_p99):
        return [
            {"overhead_us": {"median": m, "p95": m * 2, "p99": p, "max": p * 3}}
            for m, p in medians_p99
        ]

    def test_the_bound_is_the_worst_of_each_statistic(self):
        bound = overhead.upper_bound(self.runs((100, 500), (300, 400), (200, 900)))
        assert bound == {"median": 300, "p95": 600, "p99": 900, "max": 2700}

    def test_one_run_bounds_itself(self):
        assert overhead.upper_bound(self.runs((5, 9)))["p99"] == 9

    def test_a_real_measurement_has_events_in_order_and_none_arriving_early(self):
        result = overhead.measure_once(requests=6, tokens=5)
        assert result["events"] == 6 * (5 + 3)  # role, 5 text, finish, usage
        assert result["events_that_appeared_to_arrive_early"] == 0
        s = result["overhead_us"]
        assert 0 < s["median"] <= s["p95"] <= s["p99"] <= s["max"]
