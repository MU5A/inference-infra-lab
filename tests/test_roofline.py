"""The roofline calculator: expected values are worked out by hand in the comments."""

from __future__ import annotations

import copy
import json

import pytest

from inferlab import roofline as rl


def roof(peak=2e12, bw=1e11):
    return rl.Roof("test", peak, bw)  # ridge = 20 FLOP per byte


class TestRoof:
    def test_ridge_is_compute_over_bandwidth(self):
        assert roof().ridge == pytest.approx(20.0)

    def test_below_the_ridge_bandwidth_sets_the_limit(self):
        assert roof().attainable(5.0) == pytest.approx(5e11)  # 1e11 * 5

    def test_at_and_above_the_ridge_compute_sets_the_limit(self):
        assert roof().attainable(20.0) == pytest.approx(2e12)
        assert roof().attainable(500.0) == pytest.approx(2e12)

    def test_the_ridge_itself_counts_as_compute_bound(self):
        assert roof().bound(19.999) == "memory"
        assert roof().bound(20.0) == "compute"

    def test_a_roof_needs_positive_numbers(self):
        with pytest.raises(ValueError):
            rl.Roof("bad", 0.0, 1e11)
        with pytest.raises(ValueError):
            rl.Roof("bad", 1e12, -1.0)


class TestGemm:
    def test_flops_count_a_multiply_and_an_add(self):
        assert rl.gemm_flops(2, 3, 4) == 48

    def test_bytes_touch_each_matrix_once(self):
        # A is 2x4, B is 4x3, C is 2x3. Two-byte inputs: (8 + 12) * 2 = 40, output 6 * 2 = 12.
        assert rl.gemm_bytes(2, 3, 4, in_bytes=2) == 52
        # Four-byte outputs: 40 + 6 * 4 = 64.
        assert rl.gemm_bytes(2, 3, 4, in_bytes=2, out_bytes=4) == 64

    def test_a_square_product_has_intensity_n_over_three_at_equal_widths(self):
        # 2 n^3 / (3 n^2 w) is n / (1.5 w); with w = 2 bytes that is n / 3.
        assert rl.gemm_intensity(4096, 4096, 4096, 2) == pytest.approx(4096 / 3)

    def test_a_decode_step_does_about_one_flop_per_byte_at_half_precision(self):
        # One row times an 8192 x 8192 half-precision matrix:
        # 134,217,728 FLOPs over 134,250,496 bytes.
        assert rl.gemm_intensity(1, 8192, 8192, 2) == pytest.approx(0.99976, abs=1e-4)


class TestKnee:
    def test_with_only_weights_it_is_ridge_times_weight_bytes_over_two(self):
        assert rl.knee_batch(8192, 8192, 2, 0, 0, ridge=30.0) == pytest.approx(30.0)

    def test_activation_traffic_pushes_the_knee_out(self):
        # 30 * 8192^2 * 2 / (2 * 8192^2 - 30 * 8192 * 2 - 30 * 8192 * 2) = 30.2214
        knee = rl.knee_batch(8192, 8192, 2, 2, 2, ridge=30.0)
        assert knee == pytest.approx(30.2214, abs=1e-3)

    def test_at_the_knee_the_intensity_equals_the_ridge(self):
        knee = rl.knee_batch(8192, 8192, 2, 2, 2, ridge=30.0)
        assert rl.gemm_intensity(knee, 8192, 8192, 2) == pytest.approx(30.0, rel=1e-9)

    def test_a_ridge_the_shape_can_never_reach_gives_none(self):
        # 2 * 64 - ridge * (8 * 2 + 8 * 2) = 128 - 32 * ridge: zero at ridge 4, so never reached.
        assert rl.knee_batch(8, 8, 2, 2, 2, ridge=4.0) is None
        assert rl.knee_batch(8, 8, 2, 2, 2, ridge=3.9) is not None


class TestPlace:
    def test_fraction_is_achieved_over_attainable(self):
        placed = rl.place(roof(), 5.0, 4e11)  # attainable 5e11
        assert placed["fraction_of_roof"] == pytest.approx(0.8)
        assert placed["bound"] == "memory" and not placed["above_roof"]

    def test_a_point_slightly_over_the_roof_is_within_tolerance(self):
        assert not rl.place(roof(), 5.0, 5e11 * 1.049)["above_roof"]
        assert rl.place(roof(), 5.0, 5e11 * 1.051)["above_roof"]


def run(best_s, flops=2e9, moved=1.2e10, validated=True):
    return {"triad": {"flops": flops, "bytes": moved, "best_s": best_s, "validated": validated}}


class TestSummarizeConfig:
    def test_rates_are_work_over_the_fastest_time_with_the_median_over_runs(self):
        s = rl.summarize_config([run(0.10), run(0.12), run(0.11)], "triad", "bandwidth")
        # bytes/s are 1.2e11, 1.0e11, 1.0909e11: the median is the middle one.
        assert s["bytes_per_s"]["median"] == pytest.approx(1.2e10 / 0.11)
        assert s["flops_per_s"]["median"] == pytest.approx(2e9 / 0.11)
        assert s["intensity"] == pytest.approx(2e9 / 1.2e10)

    def test_runs_that_disagree_are_marked_unstable(self):
        assert not rl.summarize_config([run(0.1), run(0.2), run(0.1)], "triad", "bandwidth")[
            "stable"
        ]
        assert rl.summarize_config([run(0.10), run(0.101), run(0.1)], "triad", "bandwidth")[
            "stable"
        ]

    def test_the_stability_check_follows_the_kind_asked_for(self):
        # Identical bytes and flops per call, so both kinds see the same variation here.
        runs = [run(0.1), run(0.1), run(0.1)]
        assert rl.summarize_config(runs, "triad", "compute")["stable"]

    def test_a_failed_validation_in_any_run_is_carried_through(self):
        runs = [run(0.1), run(0.1, validated=False), run(0.1)]
        assert not rl.summarize_config(runs, "triad", "bandwidth")["validated"]


class TestPredictions:
    def test_bandwidth_range_is_inclusive_at_both_ends(self):
        assert rl.check_bandwidth(60.0) == [] and rl.check_bandwidth(100.0) == []
        assert rl.check_bandwidth(59.9) and rl.check_bandwidth(100.1)

    def test_matmul_passes_when_all_three_hold(self):
        assert rl.check_matmul(2.5, 2.4, 1.5) == []

    def test_matmul_range_is_checked_at_both_ends(self):
        assert rl.check_matmul(0.99, 2.0, 0.5) and rl.check_matmul(4.01, 4.0, 1.0)
        assert rl.check_matmul(1.0, 1.0, 0.5) == [] and rl.check_matmul(4.0, 4.0, 1.0) == []

    def test_half_precision_may_be_ten_percent_slower_but_no_more(self):
        assert rl.check_matmul(2.0, 1.8, 1.0) == []
        assert any("fp16" in f for f in rl.check_matmul(2.0, 1.79, 1.0))

    def test_the_large_size_must_beat_the_small_one_by_twenty_percent(self):
        assert rl.check_matmul(2.4, 2.4, 2.0) == []
        assert any("n=512" in f for f in rl.check_matmul(2.39, 2.4, 2.0))


def point(batch, intensity, flops, bytes_per_s):
    return {
        "batch": batch,
        "intensity": intensity,
        "flops_per_s": flops,
        "bytes_per_s": bytes_per_s,
    }


class TestSkinny:
    # Ridge 20, knee 20: memory region is batch <= 5 (needs 6e10 B/s), compute region is
    # batch >= 80 (needs 1e12 FLOP/s).
    def test_good_points_pass(self):
        pts = [point(1, 1.0, 8e10, 8e10), point(80, 80.0, 1.5e12, 4e10)]
        assert rl.check_skinny(pts, roof(), knee=20.0) == []

    def test_a_point_above_its_roofline_is_a_failure(self):
        pts = [point(1, 1.0, 1.2e11, 1.2e11)]  # roofline at intensity 1 is 1e11
        assert any("above" in f for f in rl.check_skinny(pts, roof(), knee=20.0))

    def test_a_slow_memory_bound_point_fails_at_the_boundary_batch(self):
        assert rl.check_skinny([point(5, 5.0, 3e11, 6e10)], roof(), knee=20.0) == []
        slow = rl.check_skinny([point(5, 5.0, 2.9e11, 5.9e10)], roof(), knee=20.0)
        assert any("memory-bound" in f for f in slow)

    def test_a_point_past_the_memory_region_is_not_held_to_the_bandwidth_floor(self):
        assert rl.check_skinny([point(6, 6.0, 1e11, 1e10)], roof(), knee=20.0) == []

    def test_a_slow_compute_bound_point_fails_at_the_boundary_batch(self):
        assert rl.check_skinny([point(80, 80.0, 1e12, 1e10)], roof(), knee=20.0) == []
        slow = rl.check_skinny([point(80, 80.0, 0.99e12, 1e10)], roof(), knee=20.0)
        assert any("compute-bound" in f for f in slow)

    def test_a_point_just_under_the_compute_region_is_not_held_to_the_compute_floor(self):
        assert rl.check_skinny([point(79, 79.0, 5e11, 1e10)], roof(), knee=20.0) == []

    def test_without_a_knee_only_the_ceiling_is_checked(self):
        pts = [point(1, 1.0, 1e9, 1e9), point(512, 512.0, 1e9, 1e9)]
        assert rl.check_skinny(pts, roof(), knee=None) == []


def specs():
    fig = {
        "value": 3.35,
        "unit": "TB/s",
        "how": "printed",
        "as_printed": "3.35TB/s",
        "source": "https://example.com/spec",
        "fetched": "2026-11-23",
    }
    derived = {
        "value": 989.5,
        "unit": "TFLOPS",
        "how": "derived",
        "from": "1,979 TFLOPS printed with sparsity",
        "rule": "dense is half of sparse",
        "source": "https://example.com/spec",
        "fetched": "2026-11-23",
    }
    return {
        "fetched": "2026-11-23",
        "label": "L",
        "devices": [{"name": "Dev", "memory_bandwidth": fig, "compute": {"fp16_dense": derived}}],
    }


class TestSpecs:
    def test_a_complete_file_has_no_problems(self):
        assert rl.validate_specs(specs()) == []

    @pytest.mark.parametrize(
        "mutate, expect",
        [
            (lambda d: d["devices"][0]["memory_bandwidth"].pop("source"), "no source URL"),
            (
                lambda d: d["devices"][0]["memory_bandwidth"].update(fetched="yesterday"),
                "fetch date",
            ),
            (lambda d: d["devices"][0]["memory_bandwidth"].update(unit="furlongs"), "unit"),
            (lambda d: d["devices"][0]["memory_bandwidth"].update(value=0), "positive"),
            (
                lambda d: d["devices"][0]["memory_bandwidth"].update(how="guessed"),
                "printed or derived",
            ),
            (lambda d: d["devices"][0]["memory_bandwidth"].pop("as_printed"), "as printed"),
            (lambda d: d["devices"][0]["compute"]["fp16_dense"].pop("rule"), "the rule"),
            (lambda d: d["devices"][0]["compute"]["fp16_dense"].pop("from"), "came from"),
            (lambda d: d.update(label="M"), "labelled L"),
            (lambda d: d.pop("fetched"), "no fetch date"),
            (lambda d: d["devices"][0].pop("memory_bandwidth"), "missing"),
        ],
    )
    def test_each_missing_piece_of_provenance_is_reported(self, mutate, expect):
        data = copy.deepcopy(specs())
        mutate(data)
        assert any(expect in p for p in rl.validate_specs(data))

    def test_units_convert_to_bytes_and_flops_per_second(self):
        assert rl.to_si(3.35, "TB/s") == pytest.approx(3.35e12)
        assert rl.to_si(2039, "GB/s") == pytest.approx(2.039e12)
        assert rl.to_si(1.3, "PFLOPS") == pytest.approx(1.3e15)
        assert rl.to_si(312, "TFLOPS") == pytest.approx(3.12e14)

    def test_a_device_roof_pairs_the_chosen_compute_figure_with_bandwidth(self):
        r = rl.spec_roof(specs()["devices"][0], "fp16_dense")
        assert r.bandwidth == pytest.approx(3.35e12) and r.peak_flops == pytest.approx(9.895e14)
        assert r.ridge == pytest.approx(9.895e14 / 3.35e12)

    def test_a_missing_figure_gives_no_roof_rather_than_a_guess(self):
        assert rl.spec_roof(specs()["devices"][0], "fp32") is None

    def test_loading_refuses_an_invalid_file(self, tmp_path):
        bad = specs()
        bad["devices"][0]["memory_bandwidth"].pop("source")
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="no source URL"):
            rl.load_specs(path)

    def test_loading_accepts_a_valid_file(self, tmp_path):
        path = tmp_path / "ok.json"
        path.write_text(json.dumps(specs()))
        assert rl.load_specs(path)["devices"][0]["name"] == "Dev"
