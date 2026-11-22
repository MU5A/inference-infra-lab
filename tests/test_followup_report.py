"""The follow-up tables are rendered from result files, and hide any number that is not citable."""

from __future__ import annotations

from inferlab import followup_report as fr


def row(prompt=40, cold=25.0, warm=5.0, cached=39.0, cold_stable=True, warm_stable=True):
    return {
        "prompt_tokens": prompt,
        "cold_ttft_ms": cold,
        "warm_ttft_ms": warm,
        "speedup": cold / warm,
        "cold_cached_tokens": 0,
        "warm_cached_per_request": cached,
        "cold_stable": cold_stable,
        "warm_stable": warm_stable,
    }


class TestCache:
    def test_a_stable_row_shows_its_numbers_and_the_prediction_verdict(self):
        table = fr.cache_table({"comparison": {"short": row(cold=25.0, warm=5.0)}})
        line = [x for x in table.splitlines() if x.startswith("| short")][0]
        assert "25.0" in line and "5.0" in line and "5.0x" in line and "| held |" in line
        assert "| 20.0 |" in line  # saved: cold minus warm

    def test_a_failed_prediction_on_stable_data_is_shown_as_a_finding(self):
        table = fr.cache_table({"comparison": {"short": row(cold=25.3, warm=7.6)}})
        assert "3.3x" in table and "did not hold" in table

    def test_the_speedup_boundary_is_inclusive(self):
        assert "| held |" in fr.cache_table({"comparison": {"p": row(cold=50.0, warm=10.0)}})

    def test_an_unstable_row_carries_no_numbers(self):
        table = fr.cache_table(
            {"comparison": {"long": row(prompt=1469, cold=483.4, warm=20.4, warm_stable=False)}}
        )
        line = [x for x in table.splitlines() if x.startswith("| long")][0]
        assert fr.NOT_CITABLE in line
        for number in ("483", "20.4", "23.7", "1468"):
            assert number not in line

    def test_an_unstable_cold_run_hides_the_row_too(self):
        table = fr.cache_table({"comparison": {"p": row(cold=91.8, cold_stable=False)}})
        assert fr.NOT_CITABLE in table and "91.8" not in table


def bench(tg=(152.0, 0.8, True), pp=(3000.0, 6.0, True), decode_ref=150.0, prefill_ref=3005.0):
    def block(values):
        avg, sd, stable = values
        return {"tokens_per_s": avg, "stddev": sd, "cv": sd / avg, "stable": stable}

    return {
        "summary": {"pp512": block(pp), "tg128": block(tg)},
        "harness_decode_tps_baseline": decode_ref,
        "server_prefill_tps_baseline": prefill_ref,
    }


class TestBench:
    def test_each_measurement_is_compared_with_its_own_reference(self):
        lines = fr.bench_table(bench()).splitlines()
        tg = [x for x in lines if x.startswith("| tg128")][0]
        pp = [x for x in lines if x.startswith("| pp512")][0]
        assert "harness decode baseline 150.0" in tg and "+1.3%" in tg and "10%: held" in tg
        assert "server prefill, long prompt 3005.0" in pp and "-0.2%" in pp and "15%: held" in pp

    def test_a_slow_but_steady_result_is_a_finding_not_hidden(self):
        table = fr.bench_table(bench(tg=(55.7, 0.2, True)))
        assert "-62.9%" in table and "10%: did not hold" in table

    def test_the_tolerance_is_inclusive(self):
        assert "10%: held" in fr.bench_table(bench(tg=(165.0, 0.1, True), decode_ref=150.0))
        assert "10%: did not hold" in fr.bench_table(bench(tg=(166.0, 0.1, True), decode_ref=150.0))

    def test_an_unstable_measurement_shows_its_variation_and_no_speed(self):
        table = fr.bench_table(bench(tg=(60.5, 36.1, False)))
        line = [x for x in table.splitlines() if x.startswith("| tg128")][0]
        assert fr.NOT_CITABLE in line and "59.7%" in line and "60.5" not in line


def overhead(early=0):
    def run(median):
        return {
            "overhead_us": {"n": 3450, "median": median, "p95": 450.4, "p99": 740.0, "max": 5700.0}
        }

    return {
        "runs": [run(268.0), run(271.2)],
        "upper_bound_us": {"median": 271.2, "p95": 450.4, "p99": 756.0, "max": 5728.0},
        "events_that_appeared_to_arrive_early": early,
        "client_cpu_us_per_event_profiled": 744.4,
    }


class TestOverheadAndReferences:
    def test_the_upper_bound_row_follows_the_runs(self):
        lines = fr.overhead_table(overhead()).splitlines()
        assert lines[2].startswith("| 1 | 3450 | 268 |") and lines[3].startswith("| 2 |")
        assert lines[4].startswith("| **Upper bound**") and "| 271 | 450 | 756 | 5728 |" in lines[4]

    def test_early_arrivals_and_profile_cost_are_stated(self):
        text = fr.overhead_table(overhead(early=0))
        assert "before they were sent: 0." in text and "744 us" in text

    def test_references_list_each_phase(self):
        refs = {"references": {"cold": {"before_tps": 166.1, "after_tps": 150.2}}}
        assert "| cold | 166.1 | 150.2 |" in fr.references_table(refs)
