"""Predictions and limits for Day 1's three follow-up measurements. Written BEFORE they were run.

1. Cold versus warm prompt cache, in one session. A repeated prompt reuses the server's cached
   prefix, so time to first token should collapse. Predictions: warm time to first token is at
   least 5 times lower than cold for every prompt, a cold run sees no cached tokens, and a warm run
   reuses every prompt token but one (the last token is always processed again, which is what the
   rejected first attempt's counts suggested: 39 of 40, 256 of 257, 1,468 of 1,469).

2. llama-bench cross-check. llama-bench times the model with no HTTP server and an empty context, so
   it should be a little faster than the harness through the server, not far from it. Predictions:
   decode within 10% of the harness baseline and prompt processing within 15% of the server's own
   prefill rate on the longest prompt. A repeat must also pass the 5% variation limit.

3. Harness overhead is reported as a one-sided upper bound: interference only ever adds error, so
   the worst of several runs bounds what the harness itself adds. No pass or fail.
"""

from __future__ import annotations

from collections.abc import Sequence

CACHE_MIN_SPEEDUP = 5.0
DECODE_TOLERANCE = 0.10
PREFILL_TOLERANCE = 0.15
MAX_BENCH_CV = 0.05


def compare_cache(cold: dict, warm: dict) -> dict:
    """Per prompt: cold and warm time to first token, the speed-up, and cached tokens reused."""
    out = {}
    for name, cold_block in cold["results"].items():
        warm_block = warm["results"][name]
        cs, ws = cold_block["summary"], warm_block["summary"]
        ok_requests = ws["requests"] - ws["failed"]
        out[name] = {
            "prompt_tokens": cs["prompt_tokens"][0],
            "cold_ttft_ms": cs["ttft_s"]["median"] * 1e3,
            "warm_ttft_ms": ws["ttft_s"]["median"] * 1e3,
            "speedup": cs["ttft_s"]["median"] / ws["ttft_s"]["median"],
            "cold_cached_tokens": cs["cached_tokens_seen"],
            "warm_cached_per_request": ws["cached_tokens_seen"] / ok_requests,
            "cold_stable": cs["stable"],
            "warm_stable": ws["stable"],
        }
    return out


def check_cache(comparison: dict, min_speedup: float = CACHE_MIN_SPEEDUP) -> list[str]:
    failures = []
    for name, row in comparison.items():
        if row["cold_cached_tokens"] != 0:
            failures.append(f"{name}: the cold run reused cached tokens, so it was not cold")
        if row["speedup"] < min_speedup:
            failures.append(f"{name}: warm time to first token was not {min_speedup:g}x lower")
        if row["warm_cached_per_request"] != row["prompt_tokens"] - 1:
            failures.append(f"{name}: the warm run did not reuse all but one of the prompt tokens")
        if not (row["cold_stable"] and row["warm_stable"]):
            failures.append(f"{name}: the runs were not repeatable enough to cite")
    return failures


def summarize_llama_bench(rows: Sequence[dict], max_cv: float = MAX_BENCH_CV) -> dict:
    """llama-bench rows keyed by what they measure, with their own variation and a verdict."""
    out = {}
    for row in rows:
        kind = f"pp{row['n_prompt']}" if row["n_gen"] == 0 else f"tg{row['n_gen']}"
        cv = row["stddev_ts"] / row["avg_ts"] if row["avg_ts"] else None
        out[kind] = {
            "tokens_per_s": row["avg_ts"],
            "stddev": row["stddev_ts"],
            "cv": cv,
            "samples": row.get("samples_ts", []),
            "stable": cv is not None and cv <= max_cv,
        }
    return out


def check_llama_bench(
    summary: dict, harness_decode_tps: float, server_prefill_tps: float
) -> list[str]:
    failures = [
        f"{k}: variation {v['cv']:.1%} is above the limit"
        for k, v in summary.items()
        if not v["stable"]
    ]
    tg = next((v for k, v in summary.items() if k.startswith("tg")), None)
    pp = next((v for k, v in summary.items() if k.startswith("pp")), None)
    if tg and abs(tg["tokens_per_s"] - harness_decode_tps) / harness_decode_tps > DECODE_TOLERANCE:
        failures.append("decode was not within 10% of the harness baseline")
    if pp and abs(pp["tokens_per_s"] - server_prefill_tps) / server_prefill_tps > PREFILL_TOLERANCE:
        failures.append("prompt processing was not within 15% of the server's prefill rate")
    return failures
