"""Day 5 measurement: the memory llama.cpp reports for Qwen2.5-0.5B-Instruct Q4_K_M.

Every configuration (a context length and a KV cache type) is launched as its own llama-server
process, three times, interleaved so no configuration gets a different part of the day. A launch
waits until the server answers /health, is stopped with SIGINT so that it prints its memory
breakdown on the way out, and its JSON log is parsed (`inferlab.llama_log`). Nothing is timed: the
quantity measured is how many MiB llama.cpp says it allocated, so the decode-probe rule for timed
GPU phases does not apply, and no speed is reported.

The runner holds each process handle itself, so it only ever signals the process it started. A
launch that fails is kept in the results with its reason. Nothing is retried.

usage: python scripts/day05_run.py --out days/05-kv-cache-memory/results/llama_kv.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from inferlab import env, kvcheck, llama_log

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "llama-server"
MODEL = ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
LLAMA_DIR = ROOT / "third_party" / "llama.cpp"
MIN_FREE_BYTES = 8 * 10**9
READY_TIMEOUT_S = 90
STOP_TIMEOUT_S = 30
SAMPLE = (4096, "f16")  # one full log is kept so the parsing can be re-checked


def port_is_free(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def machine_state() -> dict:
    usage = shutil.disk_usage(ROOT)
    return {
        "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "power": env.power_source(),
        "thermal": env.thermal_state(),
        "load_average": list(os.getloadavg()),
        "memory_free_percent": env.memory_free_percent(),
        "swap_used_mb": env.swap_used_mb(),
        "disk_free_bytes": usage.free,
    }


def command(context: int, cache_type: str, port: int) -> list[str]:
    return [
        str(SERVER),
        "-m", str(MODEL),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-c", str(context),
        "--parallel", "1",
        "-ngl", "99",
        "-ctk", cache_type,
        "-ctv", cache_type,
        "-fa", "on",
        "--log-jsonl",
        "-lv", "4",
        "--no-webui",
    ]  # fmt: skip


def launch(context: int, cache_type: str, port: int, log_path: Path) -> dict:
    cmd = command(context, cache_type, port)
    record = {
        "context": context,
        "cache_type": cache_type,
        "ok": False,
        "error": None,
        "ready_s": None,
        "exit_code": None,
        "swap_used_mb_before": env.swap_used_mb(),
    }
    child_env = os.environ | {"DYLD_LIBRARY_PATH": str(SERVER.parent)}
    with log_path.open("wb") as out, log_path.with_suffix(".err").open("wb") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, env=child_env)
        started = time.monotonic()
        try:
            ready = False
            while time.monotonic() - started < READY_TIMEOUT_S:
                if proc.poll() is not None:
                    record["error"] = (
                        f"the server exited with status {proc.returncode} before it was ready"
                    )
                    break
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).read()
                    ready = True
                    break
                except OSError:
                    time.sleep(0.25)
            record["ready_s"] = round(time.monotonic() - started, 2)
            if ready:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(STOP_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    record["error"] = f"the server did not stop within {STOP_TIMEOUT_S} s of SIGINT"
            elif record["error"] is None:
                record["error"] = f"the server was not ready within {READY_TIMEOUT_S} s"
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    record["exit_code"] = proc.returncode
    if record["error"] is None and proc.returncode != 0:
        record["error"] = f"the server stopped with status {proc.returncode}"
    record["parsed"] = llama_log.parse(log_path.read_text(errors="replace").splitlines())
    record["ok"] = record["error"] is None
    record["stderr_bytes"] = log_path.with_suffix(".err").stat().st_size
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--rest", type=float, default=2.0, help="seconds between launches")
    parser.add_argument("--log-dir", type=Path, required=True, help="where the raw logs go")
    args = parser.parse_args()

    problems = []
    if not SERVER.exists() or not MODEL.exists():
        problems.append("the llama-server binary or the model file is missing")
    if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
        problems.append("less than 8 GB of disk is free")
    if not port_is_free(args.port):
        problems.append(f"port {args.port} is already in use")
    if problems:
        print("refusing to run:", "; ".join(problems), file=sys.stderr)
        return 3
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    before = machine_state()
    describe = env.describe(MODEL, LLAMA_DIR)
    launches = []
    for repeat in range(args.repeats):
        for cache_type in kvcheck.CACHE_TYPES:
            for context in kvcheck.CONTEXTS:
                name = f"{kvcheck.config_key(context, cache_type)}-r{repeat}"
                record = launch(context, cache_type, args.port, args.log_dir / f"{name}.jsonl")
                record["repeat"] = repeat
                launches.append(record)
                kv = record["parsed"]["kv"]
                print(
                    f"{name}: ok={record['ok']} kv={kv[0]['size_mib'] if kv else None} MiB "
                    f"cells={kv[0]['cells'] if kv else None}"
                    + (f" ({record['error']})" if record["error"] else ""),
                    flush=True,
                )
                if repeat == 0 and (context, cache_type) == SAMPLE:
                    sample = args.out.parent / "sample_server_log.jsonl"
                    shutil.copy(args.log_dir / f"{name}.jsonl", sample)
                time.sleep(args.rest)
    after = machine_state()
    summary = kvcheck.summarize_launches(launches)
    result = {
        "schema": 1,
        "what": "memory llama-server reports for Qwen2.5-0.5B Q4_K_M by context and cache type",
        "command_template": " ".join(command(0, "TYPE", args.port)).replace(f"{ROOT}/", ""),
        "repeats": args.repeats,
        "environment": describe,
        "machine_before": before,
        "machine_after": after,
        "launches": launches,
        "unusable": [
            {
                "config": kvcheck.config_key(x["context"], x["cache_type"]),
                "repeat": x["repeat"],
                "problems": kvcheck.launch_problems(x),
            }
            for x in launches
            if kvcheck.launch_problems(x)
        ],
        "summary": summary,
    }
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {args.out}: {len(launches)} launches, {len(result['unusable'])} unusable")
    return 0 if not result["unusable"] else 1


if __name__ == "__main__":
    sys.exit(main())
