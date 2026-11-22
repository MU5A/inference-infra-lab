"""Percentiles and summaries, checked against NumPy and against hand-worked cases."""

from __future__ import annotations

import random

import numpy as np
import pytest

from inferlab.stats import percentile, summarize

QS = [0, 1, 5, 25, 50, 75, 90, 95, 99, 99.9, 100]


@pytest.mark.parametrize("seed", range(30))
def test_matches_numpy_on_random_samples(seed):
    rng = random.Random(seed)
    values = [rng.uniform(0, 1000) for _ in range(rng.randint(1, 250))]
    for q in QS:
        assert percentile(values, q) == pytest.approx(float(np.percentile(values, q)), rel=1e-12)


def test_hand_worked_cases():
    assert percentile([10, 20, 30, 40], 50) == 25.0  # halfway between the middle two
    assert percentile([10, 20, 30, 40], 0) == 10.0
    assert percentile([10, 20, 30, 40], 100) == 40.0
    assert percentile([1, 2, 3, 4, 5], 75) == 4.0
    assert percentile([0, 100], 95) == pytest.approx(95.0)  # interpolates, does not round


def test_a_single_value_is_every_percentile():
    assert all(percentile([7.5], q) == 7.5 for q in QS)


def test_order_of_the_input_does_not_matter_and_the_input_is_not_changed():
    values = [5, 1, 4, 2, 3]
    assert percentile(values, 50) == 3.0
    assert values == [5, 1, 4, 2, 3]


def test_ties_and_negative_values():
    assert percentile([2, 2, 2, 2], 99) == 2.0
    assert percentile([-5, -1, -3], 50) == -3.0


@pytest.mark.parametrize("q", [-0.1, 100.1, 1000])
def test_q_outside_the_range_is_refused(q):
    with pytest.raises(ValueError):
        percentile([1, 2, 3], q)


def test_an_empty_sample_is_refused():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_summarize_known_values():
    s = summarize([1, 2, 3, 4, 100])
    assert s["n"] == 5
    assert s["median"] == 3.0
    assert (s["min"], s["max"]) == (1.0, 100.0)
    assert s["mean"] == 22.0
    assert s["p95"] == pytest.approx(80.8)  # 4 + (100 - 4) * 0.8
    assert s["p99"] == pytest.approx(96.16)


def test_summarize_accepts_any_iterable_and_empty_is_all_none():
    assert summarize(x for x in [1, 2, 3])["n"] == 3
    empty = summarize([])
    assert empty["n"] == 0
    assert all(empty[k] is None for k in ("median", "p95", "p99", "min", "max", "mean"))


class TestCoefficientOfVariation:
    def test_known_values(self):
        from inferlab.stats import coefficient_of_variation

        assert coefficient_of_variation([10, 10, 10]) == 0.0
        assert coefficient_of_variation([8, 12]) == pytest.approx(0.2)  # stdev 2, mean 10

    def test_it_matches_numpy_on_random_samples(self):
        from inferlab.stats import coefficient_of_variation

        rng = random.Random(5)
        for _ in range(30):
            values = [rng.uniform(1, 500) for _ in range(rng.randint(2, 40))]
            expected = float(np.std(values) / np.mean(values))
            assert coefficient_of_variation(values) == pytest.approx(expected, rel=1e-12)

    def test_too_few_values_or_a_zero_mean_has_none(self):
        from inferlab.stats import coefficient_of_variation

        assert coefficient_of_variation([]) is None
        assert coefficient_of_variation([7]) is None
        assert coefficient_of_variation([-1, 1]) is None
