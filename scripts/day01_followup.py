"""Day 1's three follow-up measurements: gated, watched, and judged against predictions made first.

  overhead  the harness's own timing error, three runs, as a one-sided upper bound
  cache     cold then warm prompt cache in ONE session, against a llama-server this script starts
  bench     llama-bench cross-check, with its own variation limit

Each GPU phase (the two cache runs and llama-bench) is also gated on a short decode probe within
20% of the clean baseline (crosscheck.check_reference), re-probed every minute for up to --wait-cool
seconds before the step is refused, and bracketed by a second probe afterwards: a phase that ends
slow is marked as drifting. Sustained load heats this fanless machine and slows it for minutes.

Each step waits (bounded) for the pre-flight to pass, runs under a CPU-neighbour sampler, and is
judged by the predictions and limits in inferlab.crosscheck, which were written before the first
run. Results that are unstable are written and marked, never dropped.

usage: python scripts/day01_followup.py [--only overhead,cache,bench] [--wait-quiet 180]
exit status: 0 all steps ran and held, 3 a step was refused, 4 a step failed a limit or a prediction
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

from inferlab import crosscheck, env, overhead, runner
from inferlab.watch import Watch, wait_quiet

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "days" / "01-harness-and-baseline" / "results"
LLAMA = ROOT / "third_party" / "llama.cpp"
MODEL = ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
PROMPTS = ROOT / "prompts" / "day01.json"
URL = "http://127.0.0.1:8081"
RUNS = 3


def write(name: str, payload: dict) -> None:
    (RESULTS / name).write_text(json.dumps(payload, indent=2) + "\n")


def step_overhead(cool_s: float = 0.0) -> dict:
    runs, neighbours = [], []
    for _ in range(RUNS):
        with Watch() as watch:
            runs.append(overhead.measure_once())
        neighbours.append(watch.summary())
    profile_text, cpu_us = overhead.profile_cpu()
    (RESULTS / "harness_overhead_profile.txt").write_text(profile_text)
    early = sum(r["events_that_appeared_to_arrive_early"] for r in runs)
    result = {
        "runs": runs,
        "neighbours_per_run": neighbours,
        "upper_bound_us": overhead.upper_bound(runs),
        "client_cpu_us_per_event_profiled": cpu_us,
        "events_that_appeared_to_arrive_early": early,
        "stable": early == 0,  # the clock check is the only pass/fail: the bound needs no repeat
    }
    write("harness_overhead.json", result | {"environment": env.describe()})
    return result


def start_server() -> subprocess.Popen:
    log = (RESULTS / "server.log").open("w")
    proc = subprocess.Popen(
        [str(LLAMA / "build" / "bin" / "llama-server"), "-m", str(MODEL), "--host", "127.0.0.1",
         "--port", "8081", "-c", "4096", "-ngl", "99", "--parallel", "1"],
        stdout=log, stderr=subprocess.STDOUT,
    )  # fmt: skip
    for _ in range(90):
        try:
            if httpx.get(f"{URL}/health", timeout=2).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("llama-server did not become healthy")


def step_cache(cool_s: float = 0.0) -> dict:
    prompts = json.loads(PROMPTS.read_text())
    server = start_server()
    try:
        reports, neighbours, references = {}, {}, {}
        for label, cold in (("cold", True), ("warm", False)):
            before = settle(cool_s)  # sustained load heats the machine, so each phase is gated
            send, client = runner.http_sender(URL)
            try:
                with Watch() as watch:
                    reports[label] = runner.run_bench(
                        send, prompts, runs=5, warmup=2, max_tokens=128, cold=cold
                    )
            finally:
                client.close()
            neighbours[label] = watch.summary()
            references[label] = bracket(before, reference_decode())
    finally:
        server.terminate()
        server.wait(timeout=15)
    comparison = crosscheck.compare_cache(reports["cold"], reports["warm"])
    failures = crosscheck.check_cache(comparison)
    result = {
        "references": references,
        "drift": [p for r in references.values() for p in r["drift"]],
        "comparison": comparison,
        "prediction_failures": failures,
        "neighbours": neighbours,
        "stable": reports["cold"]["stable"] and reports["warm"]["stable"],
        "reports": reports,
    }
    write("cache_comparison.json", result | {"environment": env.describe(MODEL, LLAMA)})
    return result


def baseline_figures() -> tuple[float, float]:
    base = json.loads((RESULTS / "baseline_llama_server.json").read_text())["results"]
    decode = statistics.median(b["summary"]["decode_tps"]["median"] for b in base.values())
    prefill = statistics.median(r["server_prompt_tps"] for r in base["long"]["requests"])
    return decode, prefill


def step_bench(cool_s: float = 0.0) -> dict:
    before = settle(cool_s)
    with Watch() as watch:
        proc = subprocess.run(
            [str(LLAMA / "build" / "bin" / "llama-bench"), "-m", str(MODEL), "-p", "512",
             "-n", "128", "-r", "5", "-ngl", "99", "-o", "json"],
            capture_output=True, text=True,
        )  # fmt: skip
    if proc.returncode != 0:
        raise RuntimeError(f"llama-bench exited {proc.returncode}: {proc.stderr[-300:]}")
    references = {"bench": bracket(before, reference_decode())}
    rows = json.loads(proc.stdout)
    summary = crosscheck.summarize_llama_bench(rows)
    decode, prefill = baseline_figures()
    failures = crosscheck.check_llama_bench(summary, decode, prefill)
    result = {
        "references": references,
        "drift": references["bench"]["drift"],
        "summary": summary,
        "harness_decode_tps_baseline": decode,
        "server_prefill_tps_baseline": prefill,
        "prediction_failures": failures,
        "neighbours": watch.summary(),
        "stable": all(v["stable"] for v in summary.values()),
        "raw_rows": rows,
    }
    write("llama_bench.json", result | {"environment": env.describe(MODEL, LLAMA)})
    return result


STEPS = {"overhead": step_overhead, "cache": step_cache, "bench": step_bench}


class Refused(Exception):
    """The machine was not up to speed, so the measurement was not taken."""


def reference_decode() -> float:
    """A short decode-only llama-bench, taken just before a GPU step: is the machine up to speed?"""
    proc = subprocess.run(
        [str(LLAMA / "build" / "bin" / "llama-bench"), "-m", str(MODEL), "-p", "0", "-n", "64",
         "-r", "2", "-ngl", "99", "-o", "json"],
        capture_output=True, text=True,
    )  # fmt: skip
    if proc.returncode != 0:
        raise RuntimeError(f"llama-bench exited {proc.returncode}: {proc.stderr[-300:]}")
    return statistics.median(json.loads(proc.stdout)[0]["samples_ts"])


def wait_up_to_speed(limit_s: float) -> tuple[list[str], float | None]:
    """Re-probe decode speed every minute until it is near the baseline or `limit_s` has passed."""
    baseline = baseline_figures()[0]
    probe: dict = {}

    def check() -> list[str]:
        probe["tps"] = reference_decode()
        return crosscheck.check_reference(probe["tps"], baseline)

    return wait_quiet(limit_s, poll_s=60.0, check=check), probe.get("tps")


def settle(limit_s: float) -> float:
    """Wait for decode speed to be near the baseline; refuse the step if it never is."""
    problems, tps = wait_up_to_speed(limit_s)
    if problems:
        raise Refused("; ".join(problems))
    return tps


def bracket(before: float, after: float) -> dict:
    """Decode probes either side of a GPU phase. A phase that ended slow was heating the machine."""
    return {
        "before_tps": before,
        "after_tps": after,
        "drift": crosscheck.check_reference(after, baseline_figures()[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=",".join(STEPS))
    parser.add_argument("--wait-quiet", type=float, default=180.0)
    parser.add_argument(
        "--wait-cool", type=float, default=600.0,
        help="seconds to keep re-probing decode speed before refusing a GPU step",
    )  # fmt: skip
    parser.add_argument("--allow-noisy", action="store_true")
    args = parser.parse_args()
    status = 0
    for name in args.only.split(","):
        problems = wait_quiet(args.wait_quiet)
        if problems and not args.allow_noisy:
            print(f"{name}: refused ({'; '.join(problems)})", flush=True)
            status = status or 3
            continue
        try:
            result = STEPS[name](args.wait_cool)
        except Refused as why:
            print(f"{name}: refused ({why})", flush=True)
            status = status or 3
            continue
        held = (
            result["stable"] and not result.get("prediction_failures") and not result.get("drift")
        )
        n = result.get("neighbours") or result.get("neighbours_per_run")
        print(
            f"{name}: {'held' if held else 'DID NOT HOLD'}; "
            f"prediction failures: {result.get('prediction_failures') or 'none'}; "
            f"drift: {result.get('drift') or 'none'}; "
            f"neighbours: {n}",
            flush=True,
        )
        if not held:
            status = status or 4
    return status


if __name__ == "__main__":
    sys.exit(main())
