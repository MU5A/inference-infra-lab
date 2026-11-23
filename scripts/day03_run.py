"""Run Day 3's GPU measurements: three independent process runs per phase, each phase gated.

Each phase waits (bounded) for the pre-flight to pass, then for a short decode probe to be within
20% of the Day 1 baseline (this fanless machine slows for minutes after sustained load), runs three
worker processes under a CPU-neighbour sampler, and takes a second probe afterwards. A phase that
ends slow is marked as drifting. Unstable results are written and marked, never dropped.

usage: python scripts/day03_run.py [--only bandwidth,matmul,skinny] [--runs 3]
                                   [--wait-quiet 180] [--wait-cool 600] [--rest 0] [--allow-noisy]
--rest S idles the machine for S seconds before the first run and between runs (default 0).
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

from inferlab import env, roofline
from inferlab.watch import Watch, wait_quiet

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "days" / "03-roofline" / "results"
WORKER = ROOT / "scripts" / "day03_worker.py"
PHASES = ("bandwidth", "matmul", "skinny")
FILES = {
    "bandwidth": "gpu_bandwidth.json",
    "matmul": "gpu_matmul.json",
    "skinny": "gpu_skinny.json",
}


def run_worker(phase: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(WORKER), "--phase", phase], capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"worker {phase} exited {proc.returncode}: {proc.stderr[-400:]}")
    return json.loads(proc.stdout)


def measure(phase: str, runs: int, cool_s: float, rest_s: float = 0.0) -> dict:
    before = gate.settle(cool_s)
    raw = []
    with Watch() as watch:
        for _ in range(runs):
            if rest_s > 0:
                time.sleep(rest_s)
            raw.append(run_worker(phase))
    reference = gate.bracket(before, gate.reference_decode())
    kind = "bandwidth" if phase == "bandwidth" else "compute"
    summary = {
        key: roofline.summarize_config([r["configs"] for r in raw], key, kind)
        for key in raw[0]["configs"]
    }
    result = {
        "phase": phase,
        "mlx_version": raw[0]["mlx_version"],
        "device": raw[0]["device"],
        "references": {phase: reference},
        "drift": reference["drift"],
        "summary": summary,
        "neighbours": watch.summary(),
        "stable": all(s["stable"] for s in summary.values()),
        "validated": all(s["validated"] for s in summary.values()),
        "runs": raw,
    }
    (RESULTS / FILES[phase]).write_text(
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
        held = result["stable"] and result["validated"] and not result["drift"]
        print(
            f"{phase}: {'held' if held else 'DID NOT HOLD'}; stable: {result['stable']}; "
            f"validated: {result['validated']}; drift: {result['drift'] or 'none'}; "
            f"neighbours: {result['neighbours']}",
            flush=True,
        )
        if not held:
            status = status or 4
    return status


if __name__ == "__main__":
    sys.exit(main())
