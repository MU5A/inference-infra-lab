"""Run Day 4's Metal kernel sweeps: three independent process runs per phase, each phase gated.

The same protocol as Day 3 (`scripts/day03_run.py`): each phase waits (bounded) for the pre-flight
to pass, then for a decode probe within 20% of the Day 1 baseline, runs three worker processes
under the CPU-neighbour sampler, and takes a second probe afterwards. Results are written whether
or not they held, and say which.

usage: python scripts/day04_run.py [--only stream,occupancy,fma] [--runs 3] [--wait-quiet 180]
                                   [--wait-cool 600] [--rest 0] [--allow-noisy]
exit status: 0 all phases ran and held, 3 a phase was refused, 4 a phase did not hold
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import day01_followup as gate

from inferlab import env, metal_sweep
from inferlab.watch import Watch, wait_quiet

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "days" / "04-metal-programming" / "results"
WORKER = ROOT / "scripts" / "day04_worker.py"
PHASES = ("stream", "occupancy", "fma")
KIND = {"stream": "bandwidth", "occupancy": "bandwidth", "fma": "compute"}


def run_worker(phase: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(WORKER), "--phase", phase], capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"worker {phase} exited {proc.returncode}: {proc.stderr[-400:]}")
    return json.loads(proc.stdout)


def measure(phase: str, runs: int, cool_s: float, rest_s: float) -> dict:
    before = gate.settle(cool_s)
    raw = []
    with Watch() as watch:
        for _ in range(runs):
            if rest_s > 0:
                time.sleep(rest_s)
            raw.append(run_worker(phase))
    reference = gate.bracket(before, gate.reference_decode())
    summary = metal_sweep.summarize(raw, KIND[phase])
    result = {
        "phase": phase,
        "mlx_version": raw[0]["mlx_version"],
        "device": raw[0]["device"],
        "rest_s": rest_s,
        "references": {phase: reference},
        "drift": reference["drift"],
        "summary": summary,
        "neighbours": watch.summary(),
        "stable": all(s["stable"] for s in summary.values()),
        "validated": all(s["validated"] for s in summary.values()),
        "dispatched_as_asked": all(s["dispatched_as_asked"] for s in summary.values()),
        "runs": raw,
    }
    (RESULTS / f"{phase}.json").write_text(
        json.dumps(result | {"environment": env.describe(gate.MODEL, gate.LLAMA)}, indent=2) + "\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=",".join(PHASES))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--wait-quiet", type=float, default=180.0)
    parser.add_argument("--wait-cool", type=float, default=600.0)
    parser.add_argument("--rest", type=float, default=0.0)
    parser.add_argument("--allow-noisy", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    status = 0
    for phase in args.only.split(","):
        problems = wait_quiet(args.wait_quiet)
        if problems and not args.allow_noisy:
            print(f"{phase}: refused ({'; '.join(problems)})", flush=True)
            status = status or 3
            continue
        try:
            result = measure(phase, args.runs, args.wait_cool, args.rest)
        except gate.Refused as why:
            print(f"{phase}: refused ({why})", flush=True)
            status = status or 3
            continue
        held = (
            result["stable"]
            and result["validated"]
            and result["dispatched_as_asked"]
            and not result["drift"]
        )
        print(
            f"{phase}: {'held' if held else 'DID NOT HOLD'}; stable: {result['stable']}; "
            f"validated: {result['validated']}; as asked: {result['dispatched_as_asked']}; "
            f"drift: {result['drift'] or 'none'}; neighbours: {result['neighbours']}",
            flush=True,
        )
        if not held:
            status = status or 4
    return status


if __name__ == "__main__":
    sys.exit(main())
