"""Day 6 measurements: prefill and decode with llama-bench, and time to first token with the Day 1
harness against llama-server, for the four model files.

bench   three rounds; each round runs every model once as its own llama-bench process (prefill of
        512 tokens and decode of 128 tokens, at the start of the context and after 4,096 tokens,
        3 repetitions inside the process). Rounds interleave the models so a slow spell cannot
        land on one model only. A decode probe before every process must be within 20% of Day 1's
        baseline, and one is taken after; a rest comes first.
ttft    one llama-server per model (cold prompt cache), one warm-up then three measured requests
        for the 40, 257 and 1,469 token prompts of Day 1, bracketed by the same probe.
router  the simulation, no hardware involved.

Nothing is retried. A refused or failed step is recorded and the script exits with status 1.

usage: python scripts/day06_run.py --out days/06-prefill-decode/results [--phase bench|ttft|router]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import day01_followup as gate

from inferlab import decode_model as dm
from inferlab import env, gguf, moe_router, runner
from inferlab.watch import Watch

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
BIN = ROOT / "third_party" / "llama.cpp" / "build" / "bin"
PROMPTS = ROOT / "prompts" / "day01.json"
DEPTH = 4096
N_PROMPT = 512
N_GEN = 128
BENCH_REPS = 3
ROUNDS = 3
PORT = 8096
KEEP = (
    "n_prompt", "n_gen", "n_depth", "samples_ts", "type_k", "type_v", "flash_attn",
    "n_gpu_layers", "model_size", "model_n_params", "build_commit",
)  # fmt: skip


def model_path(key: str) -> Path:
    return MODELS_DIR / dm.MODELS[key]


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def machine_state() -> dict:
    return {
        "time": now(),
        "power": env.power_source(),
        "thermal": env.thermal_state(),
        "load_average": list(os.getloadavg()),
        "memory_free_percent": env.memory_free_percent(),
        "swap_used_mb": env.swap_used_mb(),
    }


def bench_once(path: Path) -> list[dict]:
    cmd = [
        str(BIN / "llama-bench"), "-m", str(path), "-p", str(N_PROMPT), "-n", str(N_GEN),
        "-d", f"0,{DEPTH}", "-r", str(BENCH_REPS), "-ngl", "99", "-o", "json",
    ]  # fmt: skip
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=os.environ | {"DYLD_LIBRARY_PATH": str(BIN)},
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"llama-bench exited {proc.returncode}: {proc.stderr[-300:]}")
    return [{k: e[k] for k in KEEP if k in e} for e in json.loads(proc.stdout)]


def probed(step, wait_s: float) -> tuple[object, dict]:
    """Run `step` between two decode probes; refuse if the machine never gets up to speed."""
    before = gate.settle(wait_s)
    with Watch() as watch:
        result = step()
    after = gate.reference_decode()
    return result, {"probe": gate.bracket(before, after), "neighbours": watch.summary()}


def phase_bench(rest_s: float, wait_s: float) -> dict:
    out = {
        "depth": DEPTH,
        "n_prompt": N_PROMPT,
        "n_gen": N_GEN,
        "bench_repetitions": BENCH_REPS,
        "machine_before": machine_state(),
        "models": {
            k: {"file": dm.MODELS[k], "runs": [], "raw": [], "gates": []} for k in dm.MODELS
        },
    }
    for round_number in range(ROUNDS):
        for key in dm.MODELS:
            time.sleep(rest_s)
            entries, gates = probed(lambda key=key: bench_once(model_path(key)), wait_s)
            rates = dm.bench_rates(entries, (0, DEPTH))
            record = out["models"][key]
            record["runs"].append(rates)
            record["raw"].append(entries)
            record["gates"].append(gates | {"round": round_number, "time": now()})
            print(
                f"round {round_number} {key}: "
                + " ".join(f"{k}={v:.1f}" for k, v in sorted(rates.items()))
                + (f" DRIFT {gates['probe']['drift']}" if gates["probe"]["drift"] else ""),
                flush=True,
            )
    out["machine_after"] = machine_state()
    return out


def port_is_free(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def serve_and_measure(path: Path, prompts: dict) -> dict:
    if not port_is_free(PORT):
        raise RuntimeError(f"port {PORT} is in use")
    cmd = [
        str(BIN / "llama-server"), "-m", str(path), "--host", "127.0.0.1", "--port", str(PORT),
        "-c", "4096", "-ngl", "99", "--parallel", "1",
    ]  # fmt: skip
    server = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ | {"DYLD_LIBRARY_PATH": str(BIN)},
    )
    try:
        deadline = time.monotonic() + 180
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"llama-server exited {server.returncode} before it was ready")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1).read()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("llama-server was not ready in 180 s") from None
                time.sleep(0.5)
        send, client = runner.http_sender(f"http://127.0.0.1:{PORT}")
        try:
            return runner.run_bench(send, prompts, runs=3, warmup=1, max_tokens=8)
        finally:
            client.close()
    finally:
        if server.poll() is None:
            server.send_signal(signal.SIGINT)
            try:
                server.wait(30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def phase_ttft(rest_s: float, wait_s: float) -> dict:
    prompts = json.loads(PROMPTS.read_text())
    out = {"machine_before": machine_state(), "max_tokens": 8, "models": {}}
    for key in dm.MODELS:
        time.sleep(rest_s)
        report, gates = probed(lambda key=key: serve_and_measure(model_path(key), prompts), wait_s)
        summaries = {name: block["summary"] for name, block in report["results"].items()}
        out["models"][key] = {"file": dm.MODELS[key], "prompts": summaries} | gates
        print(
            f"ttft {key}: "
            + " ".join(f"{n}={s['ttft_s']['median'] * 1000:.1f}ms" for n, s in summaries.items()),
            flush=True,
        )
    out["machine_after"] = machine_state()
    return out


def write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {path.relative_to(ROOT)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("bench", "ttft", "router", "headers"), required=True)
    parser.add_argument("--rest", type=float, default=30.0, help="seconds before each process")
    parser.add_argument("--wait", type=float, default=600.0, help="longest wait for the probe")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.phase == "router":
        write(out / "router.json", {"scenarios": moe_router.scenarios()})
        return 0
    if args.phase == "headers":
        for key in dm.MODELS:
            summary = gguf.to_summary(gguf.read_header(model_path(key)))
            sha = env.sha256_of(model_path(key))
            write(out / "gguf" / f"{key}.json", summary | {"file": dm.MODELS[key], "sha256": sha})
        return 0
    if "AC Power" not in (env.power_source() or ""):
        print("refusing: not on AC power", file=sys.stderr)
        return 3
    try:
        data = {"bench": phase_bench, "ttft": phase_ttft}[args.phase](args.rest, args.wait)
    except (gate.Refused, RuntimeError, ValueError) as why:
        print(f"FAILED {args.phase}: {why}", file=sys.stderr)
        return 1
    data["environment"] = env.describe(model_path(dm.SMALL_Q4), ROOT / "third_party" / "llama.cpp")
    write(out / f"{args.phase}.json", data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
