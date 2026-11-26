"""Day 6: judge the predictions from the committed result files and render the tables.

Everything is a pure function of committed inputs: the four GGUF header summaries, the llama-bench
runs, the time-to-first-token runs, the router simulation and Day 3's bandwidth. A measurement that
did not repeat within the stability limit is shown but never used to judge a prediction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from . import decode_model as dm
from . import moe_router
from .gguf import from_summary

PROMPT_ORDER = ("short", "medium", "long")


def md(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def model_rows(
    summaries: Mapping[str, Mapping], bench: Mapping, bandwidth: float
) -> dict[str, dict]:
    rows = {}
    depth, generated = bench["depth"], bench["n_gen"]
    for key in dm.MODELS:
        g = from_summary(summaries[key])
        rates = dm.summarize_bench(bench["models"][key]["runs"])
        near, far = dm.mid_context(0, generated), dm.mid_context(depth, generated)
        bound0 = dm.decode_bound(g, bandwidth, near)
        bound_far = dm.decode_bound(g, bandwidth, far)
        tg0, tg_far = rates["tg@0"], rates[f"tg@{depth}"]
        rows[key] = {
            "file_bytes": g.file_size,
            "weight_bytes_read": dm.weight_bytes_per_token(g),
            "kv_bytes_per_token": dm.kv_bytes_per_token(g),
            "parameters": g.parameters,
            "bytes_per_token_start": dm.bytes_per_token(g, near),
            "bytes_per_token_deep": dm.bytes_per_token(g, far),
            "bound_start": bound0,
            "bound_deep": bound_far,
            "rates": rates,
            "efficiency": tg0["median"] / bound0,
            "depth_ratio_measured": tg_far["median"] / tg0["median"],
            "depth_ratio_predicted": bound_far / bound0,
        }
    return rows


def ttft_rows(ttft: Mapping) -> dict[str, list[dict]]:
    out = {}
    for key, record in ttft["models"].items():
        rows = []
        for name in PROMPT_ORDER:
            summary = record["prompts"][name]
            tokens = summary["prompt_tokens"]
            if len(tokens) != 1:
                raise ValueError(f"{key} {name}: runs disagree on the prompt length: {tokens}")
            rows.append(
                {
                    "prompt": name,
                    "tokens": tokens[0],
                    "ttft_s": summary["ttft_s"]["median"],
                    "cv": summary["cv_ttft"],
                    "stable": summary["cv_ttft"] is not None
                    and summary["cv_ttft"] <= dm.STABLE_CV
                    and summary["failed"] == 0,
                }
            )
        out[key] = rows
    return out


def _unstable(rows: Mapping[str, dict], keys: Sequence[tuple[str, str]]) -> list[str]:
    return [
        f"not judged: {m} {k} did not repeat within {dm.STABLE_CV:.0%}"
        for m, k in keys
        if not rows[m]["rates"][k]["stable"]
    ]


def judge(rows: Mapping[str, dict], ttft: Mapping[str, list[dict]], router: list[dict], depth: int):
    models = list(dm.MODELS)
    eff = {m: rows[m]["efficiency"] for m in models}
    results = {m: rows[m]["rates"] for m in models}
    quants = (dm.SMALL_Q4, dm.SMALL_Q8, dm.SMALL_F16)
    needs = {
        "1": [(m, "tg@0") for m in models],
        "2": [(m, "tg@0") for m in models],
        "3": [(dm.SMALL_Q4, "tg@0"), (dm.LARGE_Q4, "tg@0")],
        "4": [(m, "pp@0") for m in quants] + [(dm.SMALL_Q4, "tg@0"), (dm.SMALL_F16, "tg@0")],
        "5": [(dm.SMALL_Q4, "pp@0"), (dm.LARGE_Q4, "pp@0")],
        "6": [(m, k) for m in models for k in ("tg@0", f"tg@{depth}")],
        "10": [(dm.LARGE_Q4, "tg@0"), (dm.SMALL_F16, "tg@0")],
    }
    checks = {
        "1": lambda: dm.check_bound(eff),
        "2": lambda: dm.check_efficiency(eff),
        "3": lambda: dm.check_larger_model(eff),
        "4": lambda: dm.check_prefill_compute_bound(results),
        "5": lambda: dm.check_prefill_size(results),
        "6": lambda: dm.check_depth(
            {m: rows[m]["depth_ratio_measured"] for m in models},
            {m: rows[m]["depth_ratio_predicted"] for m in models},
        ),
        "10": lambda: dm.check_same_bytes(results),
    }
    failures = {n: _unstable(rows, needs[n]) or checks[n]() for n in checks}
    shaky = [
        f"not judged: {m} {r['prompt']} prompt did not repeat within {dm.STABLE_CV:.0%}"
        for m, series in ttft.items()
        for r in series
        if not r["stable"]
    ]
    failures["7"] = shaky or dm.check_ttft(
        {m: [(r["tokens"], r["ttft_s"]) for r in series] for m, series in ttft.items()}
    )
    failures["8"] = moe_router.check_uniform(router)
    failures["9"] = moe_router.check_zipf(router)
    return failures


def analyze(
    summaries: Mapping[str, Mapping],
    bench: Mapping,
    ttft: Mapping,
    router: list[dict],
    bandwidth: float,
) -> dict:
    rows = model_rows(summaries, bench, bandwidth)
    first_token = ttft_rows(ttft)
    return {
        "bandwidth_bytes_per_s": bandwidth,
        "depth": bench["depth"],
        "n_prompt": bench["n_prompt"],
        "n_gen": bench["n_gen"],
        "models": rows,
        "ttft": {
            m: {
                "points": series,
                "ms_per_token": [
                    s * 1000 for s in dm.ttft_slopes([(r["tokens"], r["ttft_s"]) for r in series])
                ],
            }
            for m, series in first_token.items()
        },
        "router": router,
        "verdicts": dm.verdicts(judge(rows, first_token, router, bench["depth"])),
    }


# ------------------------------------------------------------------- tables


def _flag(stats: Mapping) -> str:
    return "" if stats["stable"] else " (unstable)"


def bound_table(a: Mapping) -> str:
    rows = []
    for key, r in a["models"].items():
        tg = r["rates"]["tg@0"]
        rows.append(
            (
                key,
                f"{r['file_bytes'] / 1e9:.2f}",
                f"{r['weight_bytes_read'] / 1e6:.1f}",
                f"{r['bytes_per_token_start'] / 1e6:.1f}",
                f"{r['bound_start']:.1f}",
                f"{tg['median']:.1f}{_flag(tg)}",
                f"{tg['min']:.1f} to {tg['max']:.1f}",
                f"{r['efficiency']:.2f}",
            )
        )
    headers = [
        "Model file",
        "File (GB)",
        "Weights read per token (MB)",
        "All bytes per token (MB)",
        "Bound (tokens/s)",
        "Measured decode (tokens/s)",
        "Range over runs",
        "Measured / bound",
    ]
    return md(headers, rows)


def prefill_table(a: Mapping) -> str:
    rows = []
    for key, r in a["models"].items():
        pp = r["rates"]["pp@0"]
        rows.append(
            (
                key,
                f"{r['parameters'] / 1e6:.0f}",
                f"{pp['median']:.0f}{_flag(pp)}",
                f"{pp['min']:.0f} to {pp['max']:.0f}",
                f"{pp['median'] / r['rates']['tg@0']['median']:.1f}",
            )
        )
    headers = [
        "Model file",
        "Parameters in file (M)",
        f"Prefill, {a['n_prompt']} tokens (tokens/s)",
        "Range over runs",
        "Prefill / decode",
    ]
    return md(headers, rows)


def depth_table(a: Mapping) -> str:
    depth = a["depth"]
    rows = []
    for key, r in a["models"].items():
        deep = r["rates"][f"tg@{depth}"]
        rows.append(
            (
                key,
                f"{r['kv_bytes_per_token'] / 1024:.0f}",
                f"{r['bytes_per_token_deep'] / 1e6:.1f}",
                f"{deep['median']:.1f}{_flag(deep)}",
                f"{r['depth_ratio_measured']:.3f}",
                f"{r['depth_ratio_predicted']:.3f}",
            )
        )
    headers = [
        "Model file",
        "KV per token (KiB)",
        f"Bytes per token at {depth} (MB)",
        f"Decode at {depth} (tokens/s)",
        "Deep / start, measured",
        "Deep / start, predicted",
    ]
    return md(headers, rows)


def ttft_table(a: Mapping) -> str:
    rows = []
    for key, record in a["ttft"].items():
        for r in record["points"]:
            rows.append(
                (
                    key,
                    r["tokens"],
                    f"{r['ttft_s'] * 1000:.1f}" + ("" if r["stable"] else " (unstable)"),
                )
            )
    first = md(["Model file", "Prompt tokens", "Time to first token (ms)"], rows)
    slopes = [
        (key, f"{rec['ms_per_token'][0]:.3f}", f"{rec['ms_per_token'][1]:.3f}")
        for key, rec in a["ttft"].items()
    ]
    second = md(["Model file", "ms per extra token, 40 to 257", "257 to 1,469"], slopes)
    return first + "\n\n" + second


def router_table(a: Mapping) -> str:
    rows = [
        (
            r["shape"],
            r["distribution"],
            r["batch"],
            f"{r['mean_imbalance']:.2f}",
            f"{r['min_imbalance']:.2f} to {r['max_imbalance']:.2f}",
            r["trials"],
        )
        for r in a["router"]
    ]
    return md(
        ["Shape", "Popularity", "Batch (tokens)", "Busiest / mean", "Range over trials", "Trials"],
        rows,
    )


def verdict_table(a: Mapping) -> str:
    rows = [
        (
            v["number"],
            v["prediction"],
            "yes" if v["held"] else "no",
            "held" if v["held"] else "; ".join(v["failures"]),
        )
        for v in a["verdicts"]
    ]
    return md(["#", "Prediction", "Held", "Detail"], rows)


def render(a: Mapping) -> str:
    return "\n\n".join(
        [
            "### Decode against the memory-bandwidth bound",
            f"Bandwidth: {a['bandwidth_bytes_per_s'] / 1e9:.2f} GB/s, Day 3's MLX GPU triad.",
            bound_table(a),
            "### Prefill",
            prefill_table(a),
            f"### Decode deep in the context (after {a['depth']:,} tokens)",
            depth_table(a),
            "### Time to first token against prompt length (cold prompt cache)",
            ttft_table(a),
            "### Router load imbalance (simulation, popularity assumed, no model run)",
            router_table(a),
            "### The ten predictions",
            verdict_table(a),
        ]
    )
