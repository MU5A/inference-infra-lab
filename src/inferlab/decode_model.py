"""Day 6: predict decode speed from memory bandwidth and the bytes read per token, test whether
prefill is compute-bound, and measure time to first token against prompt length.

Bound model
-----------
One decode step of one sequence reads every weight tensor once, except the input embedding when the
file has a separate output matrix (only one row of the embedding is looked up), plus the K and V of
every token already in the context:

    bytes per token (context n) = weight bytes read + n x KV bytes per token
    decode bound (tokens/s)     = memory bandwidth / bytes per token

The bandwidth is Day 3's MLX GPU triad on this chip. The KV bytes per token come from the GGUF
file's own metadata (dense grouped-query models only, f16 cache).

Predictions, written BEFORE any Day 6 timing and before the three new model files were read.
Known beforehand, so not blind: Day 1's llama-bench figures for the 0.5B Q4_K_M model (about 160
decode and 3,200 prompt tokens/s) and Day 1's time to first token for that model at 40, 257 and
1,469 prompt tokens. A smoke run of llama-bench printed only its field names, no timing.

 1. For every model the measured decode rate is below the bound.
 2. Every model reaches between 0.55 and 0.95 of its bound.
 3. The 1.5B Q4_K_M reaches a higher fraction of its bound than the 0.5B Q4_K_M, because work
    done once per step weighs less when there is more to read.
 4. Prefill is compute-bound: across the 0.5B Q4_K_M, Q8_0 and FP16 files the highest prefill
    rate is less than 1.3 times the lowest, while Q4_K_M decodes at least 1.8 times as fast as
    FP16.
 5. The 1.5B Q4_K_M prefills at between 0.25 and 0.40 of the 0.5B Q4_K_M's rate (compute grows
    with parameters, about 3.1 times).
 6. At a context of 4,096 tokens decode slows by the extra KV bytes: for every model the measured
    rate relative to the start is within 0.07 of bound(4096) / bound(0).
 7. Time to first token grows linearly with prompt length: for every model the time per extra
    token between the 257 and 1,469 token prompts is within 25% of that between 40 and 257.

Router simulation (label S; not blind, I estimated both by hand):
 8. 64 experts, 8 chosen per token, uniform routing: the busiest expert carries at least 2.5 times
    the mean load at batch 8 and at most 1.15 times at batch 4,096.
 9. With Zipf(1) popularity the busiest expert carries at least twice the mean even at batch 4,096.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence

from .gguf import GGUF
from .stats import coefficient_of_variation

EMBEDDING = "token_embd.weight"
OUTPUT = "output.weight"
F16_BYTES = 2
STABLE_CV = 0.10
RUNS_REQUIRED = 3

# Prediction thresholds, fixed before the first timing.
EFFICIENCY_RANGE = (0.55, 0.95)
PREFILL_SPREAD_LIMIT = 1.3
DECODE_Q4_OVER_FP16_MIN = 1.8
PREFILL_1P5B_RANGE = (0.25, 0.40)
DEPTH_TOLERANCE = 0.07
TTFT_SLOPE_TOLERANCE = 0.25

SMALL_Q4 = "qwen2.5-0.5b-q4_k_m"
SMALL_Q8 = "qwen2.5-0.5b-q8_0"
SMALL_F16 = "qwen2.5-0.5b-fp16"
LARGE_Q4 = "qwen2.5-1.5b-q4_k_m"
MODELS = {
    SMALL_Q4: "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    SMALL_Q8: "qwen2.5-0.5b-instruct-q8_0.gguf",
    SMALL_F16: "qwen2.5-0.5b-instruct-fp16.gguf",
    LARGE_Q4: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
}


# ------------------------------------------------------------ the bound model


def _meta_int(g: GGUF, key: str) -> int:
    value = g.metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"metadata {key!r} must be a positive integer, got {value!r}")
    return value


def kv_bytes_per_token(g: GGUF) -> int:
    """K plus V bytes cached per token, f16, from the file's own metadata."""
    arch = g.metadata.get("general.architecture")
    if arch != "qwen2":
        raise ValueError(f"only qwen2 files are understood, not {arch!r}")
    layers = _meta_int(g, f"{arch}.block_count")
    heads = _meta_int(g, f"{arch}.attention.head_count")
    kv_heads = _meta_int(g, f"{arch}.attention.head_count_kv")
    embedding = _meta_int(g, f"{arch}.embedding_length")
    if embedding % heads:
        raise ValueError("embedding length is not divisible by the head count")
    return 2 * layers * kv_heads * (embedding // heads) * F16_BYTES


def weight_bytes_per_token(g: GGUF) -> int:
    """Weight bytes one decode step reads. With a separate output matrix the input embedding is
    only indexed (one row), so it is left out; without one the embedding is also the output."""
    skip = {EMBEDDING} if OUTPUT in g.by_name else set()
    if EMBEDDING not in g.by_name:
        raise ValueError("no token embedding in the file")
    return sum(t.nbytes for t in g.tensors if t.name not in skip)


def bytes_per_token(g: GGUF, context: float) -> float:
    if context < 0:
        raise ValueError("context cannot be negative")
    return weight_bytes_per_token(g) + context * kv_bytes_per_token(g)


def decode_bound(g: GGUF, bandwidth: float, context: float) -> float:
    if not bandwidth > 0:
        raise ValueError("bandwidth must be positive")
    return bandwidth / bytes_per_token(g, context)


def mid_context(depth: int, generated: int) -> float:
    """Average number of tokens already in the context while `generated` tokens are decoded."""
    return depth + generated / 2


# ------------------------------------------------------------ summaries


def summarize_rate(values: Sequence[float]) -> dict:
    """Median across independent process runs, and whether they repeat well enough to cite."""
    values = list(values)
    cv = coefficient_of_variation(values) if len(values) > 1 else None
    return {
        "median": statistics.median(values) if values else None,
        "min": min(values, default=None),
        "max": max(values, default=None),
        "cv": cv,
        "runs": len(values),
        "stable": len(values) >= RUNS_REQUIRED and cv is not None and cv <= STABLE_CV,
    }


def bench_rates(entries: Sequence[Mapping], depths: Sequence[int]) -> dict[str, float]:
    """One llama-bench invocation: the median sample of every prefill and decode entry, keyed
    like `pp@0`, `tg@4096`. Refuses an output that lacks one of them."""
    rates: dict[str, float] = {}
    for entry in entries:
        kind = "pp" if entry["n_prompt"] and not entry["n_gen"] else "tg"
        samples = entry["samples_ts"]
        if not samples:
            raise ValueError(f"no samples for {kind} at depth {entry['n_depth']}")
        rates[f"{kind}@{entry['n_depth']}"] = statistics.median(samples)
    missing = [f"{k}@{d}" for d in depths for k in ("pp", "tg") if f"{k}@{d}" not in rates]
    if missing:
        raise ValueError(f"llama-bench output lacks {', '.join(missing)}")
    return rates


def summarize_bench(runs: Sequence[Mapping[str, float]]) -> dict[str, dict]:
    return {key: summarize_rate([r[key] for r in runs]) for key in runs[0]}


# ------------------------------------------------------------ the predictions


def ttft_slopes(points: Sequence[tuple[float, float]]) -> list[float]:
    """Seconds per extra prompt token between consecutive (tokens, seconds) points."""
    ordered = sorted(points)
    if len(ordered) < 2 or len({n for n, _ in ordered}) != len(ordered):
        raise ValueError("need at least two different prompt lengths")
    return [(t2 - t1) / (n2 - n1) for (n1, t1), (n2, t2) in zip(ordered, ordered[1:], strict=False)]


def _med(results: Mapping, model: str, key: str) -> float:
    return results[model][key]["median"]


def check_bound(efficiency: Mapping[str, float]) -> list[str]:
    """Prediction 1."""
    return [
        f"{m}: measured {e:.2f} of the bound, at or above it"
        for m, e in efficiency.items()
        if e >= 1
    ]


def check_efficiency(efficiency: Mapping[str, float]) -> list[str]:
    """Prediction 2."""
    low, high = EFFICIENCY_RANGE
    return [
        f"{m}: {e:.2f} of the bound, outside {low} to {high}"
        for m, e in efficiency.items()
        if not low <= e <= high
    ]


def check_larger_model(efficiency: Mapping[str, float]) -> list[str]:
    """Prediction 3."""
    big, small = efficiency[LARGE_Q4], efficiency[SMALL_Q4]
    if big > small:
        return []
    return [f"1.5B reaches {big:.2f} of its bound, 0.5B reaches {small:.2f}"]


def check_prefill_compute_bound(results: Mapping) -> list[str]:
    """Prediction 4."""
    failures = []
    prefill = [_med(results, m, "pp@0") for m in (SMALL_Q4, SMALL_Q8, SMALL_F16)]
    spread = max(prefill) / min(prefill)
    if not spread < PREFILL_SPREAD_LIMIT:
        failures.append(
            f"prefill rates differ {spread:.2f} times, not under {PREFILL_SPREAD_LIMIT}"
        )
    ratio = _med(results, SMALL_Q4, "tg@0") / _med(results, SMALL_F16, "tg@0")
    if ratio < DECODE_Q4_OVER_FP16_MIN:
        failures.append(
            f"Q4_K_M decodes {ratio:.2f} times as fast as FP16, under {DECODE_Q4_OVER_FP16_MIN}"
        )
    return failures


def check_prefill_size(results: Mapping) -> list[str]:
    """Prediction 5."""
    ratio = _med(results, LARGE_Q4, "pp@0") / _med(results, SMALL_Q4, "pp@0")
    low, high = PREFILL_1P5B_RANGE
    if low <= ratio <= high:
        return []
    return [f"1.5B prefills at {ratio:.2f} of the 0.5B rate, outside {low} to {high}"]


def check_depth(measured: Mapping[str, float], predicted: Mapping[str, float]) -> list[str]:
    """Prediction 6: decode(4096)/decode(0), measured against predicted."""
    return [
        f"{m}: measured ratio {measured[m]:.3f}, predicted {predicted[m]:.3f}"
        for m in measured
        if abs(measured[m] - predicted[m]) > DEPTH_TOLERANCE
    ]


def check_ttft(points: Mapping[str, Sequence[tuple[float, float]]]) -> list[str]:
    """Prediction 7."""
    failures = []
    for model, series in points.items():
        first, last = ttft_slopes(series)[0], ttft_slopes(series)[-1]
        if first <= 0 or abs(last - first) > TTFT_SLOPE_TOLERANCE * first:
            failures.append(
                f"{model}: {first * 1000:.3f} then {last * 1000:.3f} ms per extra prompt token"
            )
    return failures


PREDICTIONS = (
    ("1", "Measured decode is below the bound for every model"),
    ("2", "Every model reaches 0.55 to 0.95 of its bound"),
    ("3", "The 1.5B Q4_K_M reaches a higher fraction of its bound than the 0.5B Q4_K_M"),
    ("4", "Prefill within 1.3x across quantizations while Q4_K_M decodes 1.8x faster than FP16"),
    ("5", "1.5B prefill is 0.25 to 0.40 of 0.5B prefill"),
    ("6", "Decode at context 4,096 slows by the extra KV bytes, within 0.07, every model"),
    ("7", "Time to first token is linear in prompt length (slopes within 25%)"),
    ("8", "Uniform routing: busiest expert 2.5x the mean at batch 8, at most 1.15x at 4,096"),
    ("9", "Zipf(1) popularity: busiest expert at least 2x the mean even at batch 4,096"),
)  # fmt: skip


def verdicts(failures: Mapping[str, Sequence[str]]) -> list[dict]:
    """One row per prediction. A prediction nobody judged is reported as not held."""
    rows = []
    for number, text in PREDICTIONS:
        found = list(failures[number]) if number in failures else ["not judged"]
        rows.append({"number": number, "prediction": text, "held": not found, "failures": found})
    return rows
