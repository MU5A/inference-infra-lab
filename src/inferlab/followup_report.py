"""Markdown tables for Day 1's three follow-up measurements, rendered from their result files.

A row whose runs were not repeatable gets no numbers at all, only the reason, so an unstable
measurement cannot be copied out of the write-up by accident. A prediction that failed on stable
data is a finding and is shown as one.
"""

from __future__ import annotations

from . import crosscheck

NOT_CITABLE = "not citable (runs were not repeatable)"


def overhead_table(result: dict) -> str:
    lines = [
        "| Run | Events | Median us | p95 us | p99 us | Max us |",
        "|---|---|---|---|---|---|",
    ]
    for index, run in enumerate(result["runs"], start=1):
        o = run["overhead_us"]
        lines.append(
            f"| {index} | {o['n']} | {o['median']:.0f} | {o['p95']:.0f} | {o['p99']:.0f} "
            f"| {o['max']:.0f} |"
        )
    u = result["upper_bound_us"]
    lines.append(
        f"| **Upper bound** (worst of each column) | | {u['median']:.0f} | {u['p95']:.0f} "
        f"| {u['p99']:.0f} | {u['max']:.0f} |"
    )
    early = result["events_that_appeared_to_arrive_early"]
    return "\n".join(lines) + (
        f"\n\nEvents that appeared to arrive before they were sent: {early}. "
        f"Client CPU per event with the profiler attached: "
        f"{result['client_cpu_us_per_event_profiled']:.0f} us (the profiler inflates this)."
    )


def cache_table(result: dict) -> str:
    lines = [
        "| Prompt | Prompt tokens | Cold TTFT ms | Warm TTFT ms | Saved ms | Speed-up | "
        "Cached tokens reused per warm request | Predicted at least "
        f"{crosscheck.CACHE_MIN_SPEEDUP:g}x |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, row in result["comparison"].items():
        if not (row["cold_stable"] and row["warm_stable"]):
            lines.append(f"| {name} | {row['prompt_tokens']} | {NOT_CITABLE} | | | | | |")
            continue
        held = row["speedup"] >= crosscheck.CACHE_MIN_SPEEDUP
        lines.append(
            f"| {name} | {row['prompt_tokens']} | {row['cold_ttft_ms']:.1f} "
            f"| {row['warm_ttft_ms']:.1f} | {row['cold_ttft_ms'] - row['warm_ttft_ms']:.1f} "
            f"| {row['speedup']:.1f}x "
            f"| {row['warm_cached_per_request']:.0f} | {'held' if held else 'did not hold'} |"
        )
    return "\n".join(lines)


def bench_table(result: dict) -> str:
    lines = [
        "| Measurement | llama-bench tokens/s (standard deviation) | Variation | "
        "Compared with | Difference | Predicted within |",
        "|---|---|---|---|---|---|",
    ]
    for kind, row in result["summary"].items():
        if not row["stable"]:
            lines.append(f"| {kind} | {NOT_CITABLE} (variation {row['cv']:.1%}) | | | | |")
            continue
        if kind.startswith("tg"):
            ref = result["harness_decode_tps_baseline"]
            label, tol = "harness decode baseline", crosscheck.DECODE_TOLERANCE
        else:
            ref = result["server_prefill_tps_baseline"]
            label, tol = "server prefill, long prompt", crosscheck.PREFILL_TOLERANCE
        diff = (row["tokens_per_s"] - ref) / ref
        verdict = "held" if abs(diff) <= tol else "did not hold"
        lines.append(
            f"| {kind} | {row['tokens_per_s']:.1f} ({row['stddev']:.1f}) | {row['cv']:.1%} "
            f"| {label} {ref:.1f} | {diff:+.1%} | {tol:.0%}: {verdict} |"
        )
    return "\n".join(lines)


def references_table(result: dict) -> str:
    lines = [
        "| Phase | Decode probe before, tokens/s | Decode probe after, tokens/s |",
        "|---|---|---|",
    ]
    for phase, ref in result["references"].items():
        lines.append(f"| {phase} | {ref['before_tps']:.1f} | {ref['after_tps']:.1f} |")
    return "\n".join(lines)
