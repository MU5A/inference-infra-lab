"""Describe the machine and tools a measurement ran on. Nothing here may raise.

A benchmark number is only meaningful with its conditions attached, so every results file embeds
this record. Anything that cannot be read is recorded as None instead of being guessed.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


def _run(args: list[str], timeout: float = 5.0) -> str | None:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _sysctl(name: str) -> str | None:
    return _run(["sysctl", "-n", name])


def thermal_state() -> str | None:
    """The raw `pmset -g therm` text. A run that starts throttled should not be trusted."""
    return _run(["pmset", "-g", "therm"])


def power_source() -> str | None:
    text = _run(["pmset", "-g", "batt"])
    return text.splitlines()[0] if text else None


_LIMIT = re.compile(r"(CPU_Speed_Limit|CPU_Scheduler_Limit)\s*=\s*(\d+)")


def preflight(max_load_per_core: float = 0.5, require_ac: bool = True) -> list[str]:
    """Reasons this machine is not in a fit state to benchmark. An empty list means it is.

    A busy machine, a laptop on battery, or a throttled chip all change the numbers, so they are
    refused up front. Results from an overridden run carry the list, so they cannot pass as clean.
    """
    problems: list[str] = []
    cores = os.cpu_count() or 1
    try:
        load = os.getloadavg()[0]
    except OSError:
        load = None
    if load is not None and load > cores * max_load_per_core:
        problems.append(
            f"1-minute load average {load:.1f} is above {cores * max_load_per_core:.1f} "
            f"({max_load_per_core} per core)"
        )
    power = power_source()
    if require_ac and power and "Battery Power" in power:
        problems.append("running on battery power")
    thermal = thermal_state() or ""
    for name, value in _LIMIT.findall(thermal):
        if int(value) < 100:
            problems.append(f"thermal throttling: {name} = {value}")
    for line in thermal.splitlines():
        if "warning level" in line and "No " not in line:
            problems.append(f"thermal warning: {line.strip()}")
    return problems


def sha256_of(path: Path) -> str | None:
    """Prefer the published digest saved beside the file, so a 500 MB file is not rehashed."""
    sidecar = path.with_name(path.name + ".sha256")
    try:
        if sidecar.exists():
            return sidecar.read_text().split()[0]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def git_commit(directory: Path) -> str | None:
    return _run(["git", "-C", str(directory), "rev-parse", "HEAD"])


def describe(model_path: Path | None = None, llama_dir: Path | None = None) -> dict:
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1 = load5 = load15 = None
    memory = _sysctl("hw.memsize")
    return {
        "machine": {
            "chip": _sysctl("machdep.cpu.brand_string"),
            "arch": platform.machine(),
            "memory_gb": round(int(memory) / 2**30, 1) if memory and memory.isdigit() else None,
            "cpu_cores": os.cpu_count(),
            "os": platform.platform(),
        },
        "conditions": {
            "load_average_1_5_15": [load1, load5, load15],
            "thermal": thermal_state(),
            "power": power_source(),
        },
        "software": {
            "python": sys.version.split()[0],
            "llama_cpp_commit": git_commit(llama_dir) if llama_dir else None,
        },
        "model": {
            "file": model_path.name if model_path else None,
            "sha256": sha256_of(model_path) if model_path else None,
        },
    }
