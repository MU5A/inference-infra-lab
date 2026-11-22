"""Run the Day 2 measurements: memory bandwidth, latency, matrix multiply, and cold file reads.

Every group is gated on the pre-flight check (waiting a bounded time for the machine to settle),
runs three independent times, and is judged against stability limits that were fixed before the
first run (see inferlab.memory.MAX_CV). Results that are not repeatable are written and marked,
never dropped, and the exit status says so.

usage: python scripts/day02_run.py [--only bandwidth,latency,compute,cold_read] [--runs 3]
                                   [--wait-quiet 180] [--allow-noisy]
exit status: 0 all groups ran and were stable, 3 a group was refused, 4 a group was unstable
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from inferlab import env
from inferlab.memory import across_runs, check_cold_predictions, cold_vs_warm, latency_summary
from inferlab.watch import Watch, wait_quiet

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "microbench" / "build"
OUT = ROOT / "days" / "02-memory-hierarchy" / "results"
MODEL = ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
KERNELS = ("copy", "scale", "add", "triad")
# Samples per variant within one process run, of which the fastest is kept. Interference can only
# slow a sample down, so more samples make the best one robust to a neighbour that appears for a
# few seconds. Raised from 3 to 8 after two runs were each spoiled by another process's builds.
COMPUTE_REPS = 8


def run_json(args: list, extra_env: dict | None = None) -> dict:
    proc = subprocess.run(
        [str(a) for a in args],
        capture_output=True,
        text=True,
        env={**os.environ, **(extra_env or {})},
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{args[0]} exited {proc.returncode}: {proc.stderr.strip()[:300]}")
    return json.loads(proc.stdout)


def bandwidth(runs: int) -> dict:
    results = {}
    for qos, thread_counts in (("interactive", (1, 2, 3, 4, 6, 8)), ("background", (1, 4))):
        for threads in thread_counts:
            sweeps = [
                run_json([BIN / "membw", "--threads", threads, "--qos", qos]) for _ in range(runs)
            ]
            if not all(s["validated"] for s in sweeps):
                raise RuntimeError(f"STREAM arrays failed validation at {threads} threads")
            results[f"{qos}-{threads}"] = {
                "qos": qos,
                "threads": threads,
                "kernels": {
                    k: across_runs([s["kernels"][k]["gbs_median"] for s in sweeps], "bandwidth")
                    for k in KERNELS
                },
                "raw": sweeps,
            }
    stable = all(k["stable"] for r in results.values() for k in r["kernels"].values())
    return {"configs": results, "stable": stable}


def latency(runs: int) -> dict:
    out = {}
    for qos, max_mib in (("interactive", 1024), ("background", 256)):
        sweeps = [
            run_json(
                [BIN / "chase", "--per-octave", 4, "--max-mib", max_mib, "--qos", qos, "--seed", n]
            )
            for n in range(1, runs + 1)
        ]
        out[qos] = latency_summary(sweeps) | {"raw": sweeps}
    return out | {"stable": all(v["stable"] for v in out.values())}


def compute(runs: int) -> dict:
    def collect(extra_env=None, variants=None):
        args = [BIN / "matmul", "--sizes", "128,256,512,1024", "--reps", COMPUTE_REPS]
        if variants:
            args += ["--variants", variants]
        sweeps = []
        for _ in range(runs):
            with Watch() as watch:
                sweep = run_json(args, extra_env)
            sweep["neighbours"] = watch.summary()  # per run, so a drifting run can be explained
            sweeps.append(sweep)
        return sweeps

    groups = {
        "default": collect(),
        "accelerate_one_thread": collect({"VECLIB_MAXIMUM_THREADS": "1"}, "accelerate"),
    }
    summary: dict = {}
    for label, sweeps in groups.items():
        for index, size in enumerate(r["n"] for r in sweeps[0]["results"]):
            for name in sweeps[0]["results"][index]["variants"]:
                values = [s["results"][index]["variants"][name] for s in sweeps]
                if not all(v["ok"] for v in values):
                    raise RuntimeError(f"{name} at n={size} failed validation")
                summary[f"{label}/{name}/{size}"] = across_runs(
                    [v["gflops_best"] for v in values], "compute"
                )
    return {"summary": summary, "raw": groups, "stable": all(v["stable"] for v in summary.values())}


def cold_read(runs: int) -> dict:
    """First and second pass over a file that starts out of the page cache, four ways.

    Each way gets its own fresh copy, made with the cache bypassed, because a file can only be cold
    once. The copies are removed as soon as they are measured.
    """
    ways = {
        "touch": ["--mode", "touch"],
        "read": ["--mode", "read"],
        "willneed": ["--mode", "touch", "--advise", "willneed"],
        "sequential": ["--mode", "touch", "--advise", "sequential"],
    }
    trials = []
    for trial in range(1, runs + 1):
        record: dict = {"checks": {}}
        for way, flags in ways.items():
            copy = ROOT / "models" / f"day02-cold-{trial}-{way}.gguf"
            try:
                run_json([BIN / "copy_nocache", MODEL, copy])
                cold = run_json([BIN / "mmap_touch", copy, *flags])
                warm = run_json([BIN / "mmap_touch", copy, *flags])
            finally:
                subprocess.run([ROOT / "scripts" / "prune_models.sh", copy.name], check=False)
            record[f"cold_{way}"] = cold
            record[f"warm_{way}"] = warm
            record["checks"][f"cold_{way}"] = cold_vs_warm(cold["gbs"], warm["gbs"])
        trials.append(record)
    summary = {
        way: across_runs([t[f"cold_{way}"]["gbs"] for t in trials], "cold_read") for way in ways
    }
    predictions = check_cold_predictions(trials)
    really_cold = all(c["really_cold"] for t in trials for c in t["checks"].values())
    return {
        "trials": trials,
        "cold_gbs": summary,
        "prediction_failures": predictions,
        "all_really_cold": really_cold,
        "stable": all(v["stable"] for v in summary.values()) and really_cold,
    }


GROUPS = {"bandwidth": bandwidth, "latency": latency, "compute": compute, "cold_read": cold_read}


def os_reported() -> dict:
    keys = [
        "hw.perflevel0.l1dcachesize",
        "hw.perflevel0.l2cachesize",
        "hw.perflevel1.l1dcachesize",
        "hw.perflevel1.l2cachesize",
        "hw.cachelinesize",
        "hw.pagesize",
        "hw.perflevel0.physicalcpu",
        "hw.perflevel1.physicalcpu",
    ]
    found = {}
    for key in keys:
        proc = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
        found[key] = (
            int(proc.stdout) if proc.returncode == 0 and proc.stdout.strip().isdigit() else None
        )
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=",".join(GROUPS))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--wait-quiet", type=float, default=180.0)
    parser.add_argument("--allow-noisy", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT / "microbench.json")
    args = parser.parse_args()

    subprocess.run(["make", "-C", ROOT / "microbench"], check=True, capture_output=True)
    report: dict = {
        "runs_per_measurement": args.runs,
        "os_reported": os_reported(),
        "compiler": subprocess.run(
            ["cc", "--version"], capture_output=True, text=True
        ).stdout.splitlines()[0],
        "groups": {},
    }
    status = 0
    for name in args.only.split(","):
        problems = wait_quiet(args.wait_quiet)
        if problems and not args.allow_noisy:
            report["groups"][name] = {"skipped": problems}
            print(f"{name}: refused ({'; '.join(problems)})")
            status = status or 3
            continue
        started = time.monotonic()
        with Watch() as watch:
            result = GROUPS[name](args.runs)
        result["neighbours"] = watch.summary()
        result["preflight_problems_overridden"] = problems
        result["seconds"] = round(time.monotonic() - started, 1)
        report["groups"][name] = result
        peak = result["neighbours"]["peak_other_cpu_percent"]
        print(
            f"{name}: {'stable' if result['stable'] else 'UNSTABLE'} in {result['seconds']} s; "
            f"other processes peaked at {peak}% CPU",
            flush=True,
        )
        if not result["stable"]:
            status = status or 4
    report["environment"] = env.describe(MODEL)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")
    return status


if __name__ == "__main__":
    sys.exit(main())
