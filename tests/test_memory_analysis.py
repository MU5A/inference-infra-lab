"""Plateau detection, tested on curves whose true levels are known."""

from __future__ import annotations

import random

import pytest

from inferlab.memory import MAX_CV, across_runs, merge_latency_runs, plateaus

KIB = 1024
MIB = 1024 * KIB


def synthetic(levels, *, noise=0.04, seed=0, per_octave=4, lo=4 * KIB, hi=1024 * MIB):
    """A latency curve with steps at the given (capacity_bytes, ns) levels and a short ramp."""
    rng = random.Random(seed)
    curve = []
    size = float(lo)
    while size <= hi:
        ns = levels[-1][1]
        previous = None
        for capacity, level_ns in levels:
            if size <= capacity:
                ns = level_ns
                break
            previous = (capacity, level_ns)
        # one point of ramp just after each capacity, as real curves have
        if previous and size <= previous[0] * 1.5:
            ns = (previous[1] + ns) / 2
        curve.append((int(size), ns * (1 + rng.uniform(-noise, noise))))
        size *= 2 ** (1 / per_octave)
    return curve


LEVELS = [(128 * KIB, 1.6), (16 * MIB, 5.5), (64 * MIB, 25.0), (2048 * MIB, 105.0)]


def test_finds_every_level_of_a_clean_curve():
    found = plateaus(synthetic(LEVELS))
    assert len(found) == 4
    assert [round(p.median_ns) for p in found] == pytest.approx([2, 6, 25, 105], abs=2)


def test_the_end_of_each_plateau_is_at_or_just_below_the_true_capacity():
    found = plateaus(synthetic(LEVELS))
    # The last level's capacity lies beyond the sweep, so only the levels whose capacity was
    # actually crossed can be checked.
    for plateau, (capacity, _) in zip(found[:-1], LEVELS[:-1], strict=True):
        assert plateau.end_bytes <= capacity * 1.001
        assert plateau.end_bytes >= capacity / 2  # within one octave


@pytest.mark.parametrize("seed", range(10))
def test_robust_to_noise_across_seeds(seed):
    found = plateaus(synthetic(LEVELS, noise=0.08, seed=seed))
    assert len(found) == 4


def test_a_flat_curve_is_one_plateau():
    curve = [(KIB * 2**k, 3.0) for k in range(12)]
    found = plateaus(curve)
    assert len(found) == 1 and found[0].points == 12


def test_a_steady_ramp_has_no_plateau_longer_than_chance():
    curve = [(KIB * 2**k, 1.5**k) for k in range(15)]
    assert all(p.points <= 3 for p in plateaus(curve))


def test_too_few_points_gives_nothing():
    assert plateaus([(4096, 1.0), (8192, 1.0)]) == []
    assert plateaus([]) == []


def test_input_order_does_not_matter():
    curve = synthetic(LEVELS, seed=3)
    shuffled = list(curve)
    random.Random(1).shuffle(shuffled)
    assert plateaus(shuffled) == plateaus(curve)


def test_a_single_outlier_does_not_split_a_plateau_into_nothing():
    curve = [(KIB * 2**k, 2.0) for k in range(12)]
    curve[5] = (curve[5][0], 40.0)
    found = plateaus(curve)
    assert sum(p.points for p in found) >= 9


class TestMergeAndStability:
    def test_merge_takes_the_median_per_size_and_reports_variation(self):
        runs = [
            [{"bytes": 4096, "ns_median": 1.0}, {"bytes": 8192, "ns_median": 2.0}],
            [{"bytes": 4096, "ns_median": 1.2}, {"bytes": 8192, "ns_median": 2.0}],
            [{"bytes": 4096, "ns_median": 1.1}, {"bytes": 8192, "ns_median": 2.0}],
        ]
        merged = merge_latency_runs(runs)
        assert merged[0]["ns"] == pytest.approx(1.1)
        assert (merged[0]["min"], merged[0]["max"], merged[0]["runs"]) == (1.0, 1.2, 3)
        assert merged[1]["cv"] == 0.0

    def test_sizes_missing_from_some_runs_are_still_reported(self):
        merged = merge_latency_runs(
            [[{"bytes": 1, "ns_median": 1.0}], [{"bytes": 2, "ns_median": 2.0}]]
        )
        assert [m["bytes"] for m in merged] == [1, 2] and all(m["runs"] == 1 for m in merged)

    def test_across_runs_flags_unrepeatable_metrics_by_kind(self):
        assert across_runs([100, 101, 99], "bandwidth")["stable"] is True
        assert across_runs([100, 130, 80], "bandwidth")["stable"] is False
        # the same spread is acceptable for a cold disk read, which is naturally noisier
        assert across_runs([100, 130, 80], "cold_read")["stable"] is True

    def test_limits_were_fixed_before_the_data(self):
        assert MAX_CV == {"bandwidth": 0.10, "latency": 0.10, "compute": 0.10, "cold_read": 0.25}


def test_a_sweep_that_is_too_sparse_loses_a_narrow_level():
    """Why real runs sample 4 points per octave: at 2, a narrow level has too few points."""
    assert len(plateaus(synthetic(LEVELS, per_octave=4))) == 4
    assert len(plateaus(synthetic(LEVELS, per_octave=2))) == 3


class TestLatencySummary:
    def sweeps(self, seeds, noise=0.04, levels=LEVELS):
        return [
            {
                "points": [
                    {"bytes": b, "ns_median": ns}
                    for b, ns in synthetic(levels, noise=noise, seed=s)
                ]
            }
            for s in seeds
        ]

    def test_repeatable_sweeps_give_four_stable_plateaus(self):
        from inferlab.memory import latency_summary

        summary = latency_summary(self.sweeps([1, 2, 3]))
        assert summary["stable"] is True
        assert summary["plateaus_per_run"] == [4, 4, 4]
        assert len(summary["plateaus"]) == 4
        assert all(e["stable"] for e in summary["plateau_stability"])

    def test_runs_that_disagree_about_the_structure_are_unstable(self):
        from inferlab.memory import latency_summary

        collapsed = [(128 * KIB, 1.6), (2048 * MIB, 40.0)]  # one run sees only two levels
        sweeps = self.sweeps([1, 2]) + self.sweeps([3], levels=collapsed)
        summary = latency_summary(sweeps)
        assert summary["same_structure"] is False
        assert summary["stable"] is False

    def test_a_plateau_whose_latency_does_not_repeat_is_unstable(self):
        from inferlab.memory import latency_summary

        slow = [(128 * KIB, 1.6), (16 * MIB, 5.5), (64 * MIB, 25.0), (2048 * MIB, 150.0)]
        sweeps = self.sweeps([1, 2]) + self.sweeps([3], levels=slow)
        summary = latency_summary(sweeps)
        assert summary["same_structure"] is True
        assert summary["plateau_stability"][-1]["stable"] is False
        assert summary["stable"] is False

    def test_the_merged_curve_has_one_point_per_size_with_a_run_count(self):
        from inferlab.memory import latency_summary

        summary = latency_summary(self.sweeps([1, 2, 3]))
        assert all(m["runs"] == 3 for m in summary["merged"])

    def test_no_data_is_not_stable(self):
        from inferlab.memory import latency_summary

        assert latency_summary([])["stable"] is False


class TestColdVersusWarm:
    def test_a_clearly_faster_second_pass_means_the_first_was_cold(self):
        from inferlab.memory import cold_vs_warm

        result = cold_vs_warm(cold_gbs=2.0, warm_gbs=40.0)
        assert result["really_cold"] is True and result["ratio"] == 20.0

    def test_two_similar_passes_mean_the_file_was_already_cached(self):
        from inferlab.memory import cold_vs_warm

        assert cold_vs_warm(cold_gbs=30.0, warm_gbs=35.0)["really_cold"] is False

    def test_the_boundary_and_a_zero_cold_rate(self):
        from inferlab.memory import cold_vs_warm

        assert cold_vs_warm(2.0, 3.0)["really_cold"] is True
        assert cold_vs_warm(2.0, 2.99)["really_cold"] is False
        assert cold_vs_warm(0.0, 5.0)["really_cold"] is True


class TestColdPredictions:
    def trial(self, touch=0.07, read=1.5, willneed=0.4, sequential=0.1, cold=True):
        check = {"really_cold": cold}
        return {
            "cold_touch": {"gbs": touch},
            "cold_read": {"gbs": read},
            "cold_willneed": {"gbs": willneed},
            "cold_sequential": {"gbs": sequential},
            "checks": {
                n: dict(check)
                for n in ("cold_touch", "cold_read", "cold_willneed", "cold_sequential")
            },
        }

    def test_predictions_that_hold_have_no_failures(self):
        from inferlab.memory import check_cold_predictions

        assert check_cold_predictions([self.trial(), self.trial()]) == []

    def test_a_cold_read_that_is_not_much_faster_is_a_finding(self):
        from inferlab.memory import check_cold_predictions

        failures = check_cold_predictions([self.trial(read=0.2)])
        assert any("5x faster" in f for f in failures)

    def test_a_hint_that_does_not_help_is_a_finding(self):
        from inferlab.memory import check_cold_predictions

        failures = check_cold_predictions([self.trial(willneed=0.1)])
        assert any("willneed" in f for f in failures)

    def test_a_pass_that_was_not_cold_is_a_finding(self):
        from inferlab.memory import check_cold_predictions

        failures = check_cold_predictions([self.trial(cold=False)])
        assert len([f for f in failures if "not clearly colder" in f]) == 4

    def test_the_boundaries_are_inclusive(self):
        from inferlab.memory import check_cold_predictions

        assert check_cold_predictions([self.trial(touch=0.25, read=1.25, willneed=0.75)]) == []
