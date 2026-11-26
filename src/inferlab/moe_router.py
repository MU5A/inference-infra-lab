"""Day 6 (label S): how unevenly a mixture-of-experts router spreads a batch of tokens.

Each token picks `top_k` distinct experts out of `experts`. With `uniform` routing every set is
equally likely. With `zipf` the experts have popularity 1 / rank**s and are drawn without
replacement in proportion to it (Efraimidis-Spirakis keys, u ** (1 / weight)). The popularity is
ASSUMED: no mixture-of-experts model was run, and no real router's behaviour was measured.

Imbalance is the busiest expert's load divided by the mean load. Parallel hardware that places one
expert per device runs at the pace of the busiest one, so imbalance is wasted time.
"""

from __future__ import annotations

import heapq
import random
import statistics

DISTRIBUTIONS = ("uniform", "zipf")


def popularity(experts: int, distribution: str, s: float = 1.0) -> list[float]:
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"distribution must be one of {DISTRIBUTIONS}")
    if experts < 1 or s < 0:
        raise ValueError("need at least one expert and a non-negative exponent")
    if distribution == "uniform":
        return [1.0] * experts
    return [1.0 / rank**s for rank in range(1, experts + 1)]


def route(weights: list[float], top_k: int, batch: int, rng: random.Random) -> list[int]:
    """Tokens per expert after `batch` tokens each pick `top_k` experts."""
    experts = len(weights)
    if not 1 <= top_k <= experts:
        raise ValueError("top_k must be between 1 and the number of experts")
    if batch < 1:
        raise ValueError("batch must be at least 1")
    loads = [0] * experts
    even = len(set(weights)) == 1
    inverse = [1.0 / w for w in weights]
    for _ in range(batch):
        if even:
            chosen = rng.sample(range(experts), top_k)
        else:
            chosen = heapq.nlargest(top_k, range(experts), key=lambda i: rng.random() ** inverse[i])
        for i in chosen:
            loads[i] += 1
    return loads


def imbalance(loads: list[int]) -> float:
    total = sum(loads)
    if total == 0:
        raise ValueError("no tokens were routed")
    return max(loads) / (total / len(loads))


def simulate(
    experts: int,
    top_k: int,
    batch: int,
    distribution: str,
    *,
    trials: int,
    seed: int,
    s: float = 1.0,
) -> dict:
    if trials < 1:
        raise ValueError("need at least one trial")
    rng = random.Random(seed)
    weights = popularity(experts, distribution, s)
    values = sorted(imbalance(route(weights, top_k, batch, rng)) for _ in range(trials))
    return {
        "experts": experts,
        "top_k": top_k,
        "batch": batch,
        "distribution": distribution,
        "s": s if distribution == "zipf" else None,
        "trials": trials,
        "seed": seed,
        "mean_imbalance": statistics.fmean(values),
        "min_imbalance": values[0],
        "max_imbalance": values[-1],
    }


SHAPES = {
    "OLMoE shape (64 experts, 8 per token)": (64, 8),
    "Mixtral shape (8 experts, 2 per token)": (8, 2),
}
BATCHES = (8, 64, 512, 4096)
TRIALS = 200
BIG_BATCH_TRIALS = 40
SEED = 7


def scenarios() -> list[dict]:
    rows = []
    for name, (experts, top_k) in SHAPES.items():
        for distribution in DISTRIBUTIONS:
            for batch in BATCHES:
                trials = BIG_BATCH_TRIALS if batch >= 4096 else TRIALS
                row = simulate(experts, top_k, batch, distribution, trials=trials, seed=SEED)
                rows.append({"shape": name} | row)
    return rows


def _find(rows: list[dict], experts: int, distribution: str, batch: int) -> dict:
    for row in rows:
        if (row["experts"], row["distribution"], row["batch"]) == (experts, distribution, batch):
            return row
    raise KeyError((experts, distribution, batch))


def check_uniform(rows: list[dict]) -> list[str]:
    """Prediction 8."""
    failures = []
    small = _find(rows, 64, "uniform", 8)["mean_imbalance"]
    large = _find(rows, 64, "uniform", 4096)["mean_imbalance"]
    if small < 2.5:
        failures.append(f"batch 8: {small:.2f} times the mean, under 2.5")
    if large > 1.15:
        failures.append(f"batch 4096: {large:.2f} times the mean, over 1.15")
    return failures


def check_zipf(rows: list[dict]) -> list[str]:
    """Prediction 9."""
    value = _find(rows, 64, "zipf", 4096)["mean_imbalance"]
    return [] if value >= 2 else [f"batch 4096: {value:.2f} times the mean, under 2"]
