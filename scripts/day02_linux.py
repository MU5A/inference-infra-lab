"""Run the Linux memory experiments in a memory-limited container, three times.

Each run is a fresh container with a 96 MiB limit and no swap, no network, and the repository
mounted read-only. The image is never pulled implicitly (--pull never): it must already be present.
The results are checked against the predictions in inferlab.cgroup.check_run, which were written
before the first run. A prediction that fails is reported as a finding and the exit status is 4.

usage: python scripts/day02_linux.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from inferlab import cgroup, env

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "days" / "02-memory-hierarchy" / "results" / "linux_memory.json"
IMAGE = "python:3.12-alpine"
RUNS = 3
LIMIT = "96m"


def docker(*args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


EXPERIMENTS = ("info", "page_cache", "anonymous", "mmap_read", "mmap_private_write")


def one_run() -> dict:
    """One fresh container. Whatever finished before a kill is kept, with the exit code."""
    proc = docker(
        "run", "--rm", "--pull", "never", "--network", "none",
        f"--memory={LIMIT}", f"--memory-swap={LIMIT}", "--cpus=1",
        "-v", f"{ROOT / 'src'}:/src:ro", "-v", f"{ROOT / 'scripts'}:/scripts:ro",
        "-e", "PYTHONPATH=/src", "-e", "PYTHONDONTWRITEBYTECODE=1",
        IMAGE, "python", "/scripts/linux_memory_demo.py",
    )  # fmt: skip
    out: dict = {"exit_code": proc.returncode, "stderr_tail": proc.stderr.strip()[-300:]}
    for line in proc.stdout.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[record["experiment"]] = record["result"]
    return out


def dirty_write(sync: bool) -> dict:
    """One container whose only process writes a file twice the limit. The only difference between
    the two arms is whether each chunk is forced to disk. Docker records OOMKilled itself, so the
    outcome does not depend on any of my code surviving."""
    name = f"day02-dirty-{'sync' if sync else 'nosync'}"
    docker("rm", "-f", name)
    docker(
        "run", "--name", name, "--pull", "never", "--network", "none",
        f"--memory={LIMIT}", f"--memory-swap={LIMIT}", "--cpus=1",
        "-v", f"{ROOT / 'scripts'}:/scripts:ro", "-e", "PYTHONDONTWRITEBYTECODE=1",
        IMAGE, "python", "/scripts/dirty_write.py",
        str(cgroup.DIRTY_WRITE_MIB), "sync" if sync else "nosync",
    )  # fmt: skip
    state = docker("inspect", name, "--format", "{{.State.OOMKilled}} {{.State.ExitCode}}")
    docker("rm", "-f", name)
    oom, code = state.stdout.split()
    return {"synced": sync, "oom_killed": oom == "true", "exit_code": int(code)}


def main() -> int:
    inspect = docker("image", "inspect", IMAGE, "--format", "{{index .RepoDigests 0}}")
    if inspect.returncode != 0:
        print(f"{IMAGE} is not present. Pull it first, with approval.", file=sys.stderr)
        return 2
    info = docker("info", "--format", "{{.ServerVersion}} kernel {{.KernelVersion}}").stdout.strip()
    runs = [one_run() for _ in range(RUNS)]
    failures = [cgroup.check_run(r) for r in runs]
    dirty = [{"unsynced": dirty_write(False), "synced": dirty_write(True)} for _ in range(RUNS)]
    dirty_failures = [cgroup.check_dirty_write(d["unsynced"], d["synced"]) for d in dirty]
    report = {
        "image": inspect.stdout.strip(),
        "docker": info,
        "limit": LIMIT,
        "runs": runs,
        "prediction_failures_per_run": failures,
        "dirty_write": dirty,
        "dirty_write_prediction_failures": dirty_failures,
        "all_predictions_held": not any(failures) and not any(dirty_failures),
        "preflight_problems_informational": env.preflight(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"image {report['image']}\nDocker {info}")
    for n, (run, fail) in enumerate(zip(runs, failures, strict=True), start=1):
        done = [e for e in EXPERIMENTS if e in run]
        print(f"run {n}: container exit {run['exit_code']}; finished {done}")
        for failure in fail:
            print(f"   prediction failed: {failure}")
    for n, (d, fail) in enumerate(zip(dirty, dirty_failures, strict=True), start=1):
        u, s = d["unsynced"], d["synced"]
        print(
            f"dirty write {n}: without fsync oom_killed={u['oom_killed']} exit {u['exit_code']}; "
            f"with fsync oom_killed={s['oom_killed']} exit {s['exit_code']}; "
            f"predictions failed: {fail or 'none'}"
        )
    return 0 if report["all_predictions_held"] else 4


if __name__ == "__main__":
    sys.exit(main())
