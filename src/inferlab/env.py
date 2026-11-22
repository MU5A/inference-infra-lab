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
import time
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


_FREE_PERCENT = re.compile(r"free percentage:\s*(\d+)%")
_SWAP_USED = re.compile(r"used = ([\d.]+)M")


def memory_free_percent() -> int | None:
    """macOS's own measure of how much memory is free, as the system sees it (reclaimable cache
    counts as free). Low values mean allocations will push other data out, which disturbs a
    benchmark with a large working set."""
    out = _run(["memory_pressure"], timeout=10)
    match = _FREE_PERCENT.search(out) if out else None
    return int(match.group(1)) if match else None


def swap_used_mb() -> float | None:
    """Swap in use. Informational: old pages can sit in swap without causing contention now."""
    out = _run(["sysctl", "-n", "vm.swapusage"])
    match = _SWAP_USED.search(out) if out else None
    return float(match.group(1)) if match else None


_GPU_FIELD = re.compile(r'"Device Utilization %"=(\d+)')


def gpu_utilization(samples: int = 20, interval_s: float = 0.1) -> dict | None:
    """Apple GPU device utilization in percent, sampled over about samples x interval seconds.

    Readable without privileges through ioreg. Video playback and window compositing use the GPU in
    short bursts (51% in one sample, 0% in the next), so one reading proves nothing and the peak and
    mean over a couple of seconds are returned. None means it could not be read.
    """
    readings: list[int] = []
    for _ in range(samples):
        out = _run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"], timeout=3)
        if out is None:
            return None
        match = _GPU_FIELD.search(out)
        if match:
            readings.append(int(match.group(1)))
        time.sleep(interval_s)
    if not readings:
        return None
    return {"peak": max(readings), "mean": sum(readings) / len(readings), "samples": len(readings)}


_LIMIT = re.compile(r"(CPU_Speed_Limit|CPU_Scheduler_Limit)\s*=\s*(\d+)")


def preflight(
    max_load_per_core: float = 0.5,
    require_ac: bool = True,
    max_gpu_peak: int = 25,
    max_gpu_mean: float = 5.0,
    min_free_memory_percent: int = 30,
) -> list[str]:
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
    free = memory_free_percent()
    if free is not None and free < min_free_memory_percent:
        problems.append(f"low free memory: {free}% (the minimum is {min_free_memory_percent}%)")
    gpu = gpu_utilization()
    if gpu and (gpu["peak"] > max_gpu_peak or gpu["mean"] > max_gpu_mean):
        problems.append(
            f"GPU is busy: peak {gpu['peak']}%, mean {gpu['mean']:.0f}% "
            f"over {gpu['samples']} samples"
        )
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
            "gpu_utilization": gpu_utilization(samples=5),
            "memory_free_percent": memory_free_percent(),
            "swap_used_mb": swap_used_mb(),
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
