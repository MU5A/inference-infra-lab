"""Render Day 1's results as Markdown tables, straight from the results files.

No number in the write-up is typed by hand: they all come from this output. Writes
results/baseline_table.md and, for the follow-up measurements that have result files,
results/followup_table.md.
"""

import json
import pathlib
import statistics as st

from inferlab import followup_report

DAY = pathlib.Path(__file__).resolve().parents[1] / "days" / "01-harness-and-baseline"
report = json.loads((DAY / "results" / "baseline_llama_server.json").read_text())

lines = [
    "| Prompt | Prompt tokens | TTFT median ms (min to max) | Decode tok/s median (min to max) "
    "| ITL ms p50 / p95 / p99 | Reply tokens |",
    "|---|---|---|---|---|---|",
]
agree = [
    "| Prompt | Harness decode tok/s | Server-reported decode tok/s | Difference |",
    "|---|---|---|---|",
]
prefill = [
    "| Prompt | Prompt tokens | Server-reported prefill tok/s | Implied by harness TTFT |",
    "|---|---|---|---|",
]
for name, block in report["results"].items():
    s, reqs = block["summary"], block["requests"]
    t, d, i = s["ttft_s"], s["decode_tps"], s["itl_s"]
    tokens = s["prompt_tokens"][0]
    reply = sorted({r["completion_tokens"] for r in reqs})
    lines.append(
        f"| {name} | {tokens} "
        f"| {t['median'] * 1e3:.1f} ({t['min'] * 1e3:.1f} to {t['max'] * 1e3:.1f}) "
        f"| {d['median']:.1f} ({d['min']:.1f} to {d['max']:.1f}) "
        f"| {i['median'] * 1e3:.2f} / {i['p95'] * 1e3:.2f} / {i['p99'] * 1e3:.2f} "
        f"| {reply[0]} ({reqs[0]['finish_reason']}) |"
    )
    harness = st.median(r["decode_tps"] for r in reqs)
    server = st.median(r["server_predicted_tps"] for r in reqs)
    agree.append(
        f"| {name} | {harness:.1f} | {server:.1f} | {abs(harness - server) / server * 100:.2f}% |"
    )
    server_prefill = st.median(r["server_prompt_tps"] for r in reqs)
    implied = tokens / st.median(r["ttft_s"] for r in reqs)
    prefill.append(f"| {name} | {tokens} | {server_prefill:.0f} | {implied:.0f} |")

env = report["environment"]["before"]
header = (
    f"Settings: {report['settings']['runs']} measured runs after {report['settings']['warmup']} "
    f"warm-up runs per prompt, max_tokens {report['settings']['max_tokens']}, temperature "
    f"{report['settings']['temperature']}, seed {report['settings']['seed']}, prompt cache "
    f"{'off' if report['settings']['cold_prompt_cache'] else 'on'}. "
    f"Load average at the start: {env['conditions']['load_average_1_5_15'][0]:.2f}. "
    f"{env['conditions']['power'].split(';')[0].strip()}."
)
out = "\n\n".join(
    [
        header,
        "\n".join(lines),
        "Harness against the server's own timer:",
        "\n".join(agree),
        "Prompt processing rate:",
        "\n".join(prefill),
    ]
)
(DAY / "results" / "baseline_table.md").write_text(out + "\n")
print(out)

# The follow-ups: only the result files that exist, each rendered by the tested module.
RESULTS = DAY / "results"
sections = []
if (RESULTS / "harness_overhead.json").exists():
    overhead = json.loads((RESULTS / "harness_overhead.json").read_text())
    sections += [
        "Harness overhead, three runs against a fake server with known timing:",
        followup_report.overhead_table(overhead),
    ]
if (RESULTS / "cache_comparison.json").exists():
    cache = json.loads((RESULTS / "cache_comparison.json").read_text())
    sections += [
        "Cold against warm prompt cache, one session:",
        followup_report.cache_table(cache),
        "Decode speed probes either side of each phase, to show the machine stayed up to speed:",
        followup_report.references_table(cache),
    ]
if (RESULTS / "llama_bench.json").exists():
    bench = json.loads((RESULTS / "llama_bench.json").read_text())
    sections += [
        "llama-bench cross-check:",
        followup_report.bench_table(bench),
        "Decode speed probes either side of the run:",
        followup_report.references_table(bench),
    ]
if sections:
    follow = "\n\n".join(sections)
    (RESULTS / "followup_table.md").write_text(follow + "\n")
    print()
    print(follow)
