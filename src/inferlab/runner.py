"""Run a fixed set of prompts repeatedly and summarize the latency of each."""

from __future__ import annotations

import datetime
from collections.abc import Callable
from pathlib import Path

import httpx

from . import env
from .client import StreamResult, metrics, stream_chat
from .stats import coefficient_of_variation, summarize

# Runs this repeatable are trusted. The Day 1 baseline, taken on a quiet machine, had decode
# variation of 1.8 to 3.3% and TTFT variation under 1.7%. A run disturbed by another process
# showed 5 to 7% and, in the worst case, 83%.
MAX_DECODE_CV = 0.05
MAX_TTFT_CV = 0.10

Send = Callable[[dict], StreamResult]


def build_payload(
    prompt: str, *, max_tokens: int, temperature: float, seed: int, cold: bool
) -> dict:
    """A chat request. `cold` turns off the server's prompt-prefix cache (a llama.cpp extension)
    so every run pays for prompt processing and TTFT measures a cold prefill."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    if cold:
        payload["cache_prompt"] = False
    return payload


def summarize_runs(runs: list[dict]) -> dict:
    """Pool the per-request metrics of one prompt into the numbers worth reporting."""
    ok = [r for r in runs if r["error"] is None and r["ttft_s"] is not None]
    pooled_itl = [gap for r in ok for gap in r["itl_s"]]
    cv_ttft = coefficient_of_variation(r["ttft_s"] for r in ok)
    cv_decode = coefficient_of_variation(r["decode_tps"] for r in ok if r["decode_tps"] is not None)
    stable = (cv_ttft is None or cv_ttft <= MAX_TTFT_CV) and (
        cv_decode is None or cv_decode <= MAX_DECODE_CV
    )
    return {
        "stable": stable,
        "cv_ttft": cv_ttft,
        "cv_decode": cv_decode,
        "requests": len(runs),
        "failed": len(runs) - len(ok),
        "prompt_tokens": sorted({r["prompt_tokens"] for r in ok if r["prompt_tokens"] is not None}),
        "ttft_s": summarize(r["ttft_s"] for r in ok),
        "itl_s": summarize(pooled_itl),
        "decode_tps": summarize(r["decode_tps"] for r in ok if r["decode_tps"] is not None),
        "cached_tokens_seen": sum(r["cached_tokens"] or 0 for r in ok),
        "max_unseen_tokens": max(
            (r["unseen_tokens"] for r in ok if r["unseen_tokens"] is not None), default=None
        ),
    }


def run_bench(
    send: Send,
    prompts: dict[str, str],
    *,
    runs: int = 3,
    warmup: int = 1,
    max_tokens: int = 128,
    temperature: float = 0.0,
    seed: int = 1,
    cold: bool = True,
) -> dict:
    """Send every prompt `warmup + runs` times. Warm-up requests are run and discarded.

    Returns per-request records and a per-prompt summary. Nothing is averaged across prompts,
    because prompts of different lengths measure different things.
    """
    if runs < 1:
        raise ValueError("runs must be at least 1")
    per_prompt: dict[str, dict] = {}
    for name, prompt in prompts.items():
        payload = build_payload(
            prompt, max_tokens=max_tokens, temperature=temperature, seed=seed, cold=cold
        )
        for _ in range(warmup):
            send(payload)
        measured = [metrics(send(payload)) for _ in range(runs)]
        per_prompt[name] = {"requests": measured, "summary": summarize_runs(measured)}
    return {
        "stable": all(block["summary"]["stable"] for block in per_prompt.values()),
        "stability_limits": {"max_decode_cv": MAX_DECODE_CV, "max_ttft_cv": MAX_TTFT_CV},
        "settings": {
            "runs": runs,
            "warmup": warmup,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "cold_prompt_cache": cold,
        },
        "results": per_prompt,
    }


def http_sender(base_url: str, *, timeout: float = 120.0) -> tuple[Send, httpx.Client]:
    """A `send` function over one reused connection, and the client to close afterwards."""
    client = httpx.Client(timeout=timeout)

    def send(payload: dict) -> StreamResult:
        return stream_chat(client, base_url, payload)

    return send, client


def timestamp() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def environment(model: str | None, llama_dir: str | None) -> dict:
    return env.describe(Path(model) if model else None, Path(llama_dir) if llama_dir else None)
