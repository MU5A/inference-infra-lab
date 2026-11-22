"""Wait for a quiet machine before a measurement, and watch for interference while it runs.

A pre-flight check at the start cannot see a build that begins mid-run, which is what spoiled
several Day 2 runs. `Watch` records which other processes use the CPU every few seconds during a
measurement, by executable name only, so interference can be attributed. It does not decide
whether a result is stable: the run-to-run variation does that.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable

from . import env

# Other processes using more than this much CPU in total (150 is one and a half cores) mark a run
# as contended. Interference only ever adds error, so a contended run is not wrong, only looser.
CONTENDED_CPU_PERCENT = 150.0


class Watch:
    def __init__(self, interval_s: float = 2.0, top: int = 6):
        self.interval_s = interval_s
        self.top = top
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> None:
        busy = env.busy_processes(self.top, exclude_tree_of=os.getpid())
        if busy is not None:
            self.samples.append({"load1": os.getloadavg()[0], "busy": busy})

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample()

    def __enter__(self) -> Watch:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self) -> dict:
        totals = [sum(p["cpu_percent"] for p in s["busy"]) for s in self.samples]
        names: dict[str, float] = {}
        for sample in self.samples:
            for p in sample["busy"]:
                names[p["name"]] = max(names.get(p["name"], 0.0), p["cpu_percent"])
        worst = sorted(names.items(), key=lambda kv: kv[1], reverse=True)[:4]
        peak = max(totals, default=None)
        return {
            "samples": len(self.samples),
            "peak_other_cpu_percent": peak,
            "peak_load1": max((s["load1"] for s in self.samples), default=None),
            "contended": peak is not None and peak > CONTENDED_CPU_PERCENT,
            "busiest_other_processes": [{"name": n, "peak_cpu_percent": c} for n, c in worst],
        }


def wait_quiet(
    limit_s: float,
    poll_s: float = 10.0,
    check: Callable[[], list[str]] = env.preflight,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> list[str]:
    """Poll the pre-flight until it is clean or `limit_s` has passed. Returns what is left."""
    deadline = clock() + limit_s
    while True:
        problems = check()
        if not problems or clock() >= deadline:
            return problems
        sleep(poll_s)
