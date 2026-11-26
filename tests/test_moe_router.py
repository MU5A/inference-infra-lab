"""The router simulation: counts that must add up, and inputs it must refuse."""

from __future__ import annotations

import random

import pytest

from inferlab import moe_router as mr


def test_every_token_is_sent_to_exactly_top_k_different_experts():
    for distribution in mr.DISTRIBUTIONS:
        loads = mr.route(mr.popularity(8, distribution), 2, 100, random.Random(1))
        assert sum(loads) == 200 and len(loads) == 8 and max(loads) <= 100


def test_choosing_every_expert_is_perfectly_balanced():
    for distribution in mr.DISTRIBUTIONS:
        loads = mr.route(mr.popularity(8, distribution), 8, 50, random.Random(1))
        assert loads == [50] * 8 and mr.imbalance(loads) == 1.0


def test_imbalance_is_the_busiest_over_the_mean():
    assert mr.imbalance([4, 0, 0, 0]) == 4.0
    assert mr.imbalance([2, 2]) == 1.0
    with pytest.raises(ValueError):
        mr.imbalance([0, 0])


def test_zipf_popularity_falls_with_rank_and_loads_follow_it():
    weights = mr.popularity(8, "zipf")
    assert weights == [1 / r for r in range(1, 9)]
    loads = mr.route(weights, 1, 4000, random.Random(3))
    assert loads[0] > 2 * loads[7]


def test_the_same_seed_gives_the_same_answer_and_a_different_one_does_not():
    kwargs = {"trials": 5, "seed": 11}
    one = mr.simulate(16, 4, 32, "uniform", **kwargs)
    assert one == mr.simulate(16, 4, 32, "uniform", **kwargs)
    assert one != mr.simulate(16, 4, 32, "uniform", trials=5, seed=12)
    assert one["min_imbalance"] <= one["mean_imbalance"] <= one["max_imbalance"]


def test_bigger_batches_average_out_uniform_routing():
    small = mr.simulate(64, 8, 8, "uniform", trials=50, seed=1)["mean_imbalance"]
    large = mr.simulate(64, 8, 1024, "uniform", trials=10, seed=1)["mean_imbalance"]
    assert small > 2 and large < 1.4


@pytest.mark.parametrize(
    "call",
    [
        lambda: mr.popularity(0, "uniform"),
        lambda: mr.popularity(4, "normal"),
        lambda: mr.popularity(4, "zipf", -1),
        lambda: mr.route([1.0] * 4, 0, 10, random.Random(0)),
        lambda: mr.route([1.0] * 4, 5, 10, random.Random(0)),
        lambda: mr.route([1.0] * 4, 1, 0, random.Random(0)),
        lambda: mr.simulate(4, 1, 1, "uniform", trials=0, seed=0),
    ],
)
def test_impossible_inputs_are_refused(call):
    with pytest.raises(ValueError):
        call()


def row(distribution, batch, value):
    return {"experts": 64, "distribution": distribution, "batch": batch, "mean_imbalance": value}


def test_the_router_predictions_are_judged_at_both_ends():
    good = [row("uniform", 8, 3.9), row("uniform", 4096, 1.1), row("zipf", 4096, 6.0)]
    assert mr.check_uniform(good) == [] and mr.check_zipf(good) == []
    assert len(mr.check_uniform([row("uniform", 8, 2.0), row("uniform", 4096, 1.2)])) == 2
    assert mr.check_zipf([row("zipf", 4096, 1.9)])
    with pytest.raises(KeyError):
        mr.check_zipf([])


def test_the_scenario_list_covers_both_shapes_distributions_and_batches(monkeypatch):
    monkeypatch.setattr(mr, "TRIALS", 2)
    monkeypatch.setattr(mr, "BIG_BATCH_TRIALS", 1)
    monkeypatch.setattr(mr, "BATCHES", (8, 64))
    rows = mr.scenarios()
    assert len(rows) == 2 * 2 * 2
    assert {(r["experts"], r["top_k"]) for r in rows} == {(64, 8), (8, 2)}
