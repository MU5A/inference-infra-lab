"""How much does the benchmark client itself add to what it measures?

A fake server, in a separate process as a real server would be, stamps every streamed event with
the monotonic time at which it was about to be written. The client records when it saw the same
event. The difference is the harness's measurement error for that event: loopback delivery, HTTP and
line parsing, JSON decoding, and interpreter scheduling. Both processes read the same system-wide
monotonic clock, which is checked by confirming that no event ever appears to arrive before it was
sent.

The send timestamp is taken before the server serializes and writes each event, so server-side work
is included. Several independent runs are combined into a one-sided upper bound, the worst value of
each statistic, because interference can only add to the error.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import subprocess
import sys
import time
from collections.abc import Sequence

import httpx

from .client import stream_chat
from .stats import summarize


def _start_server(tokens: int) -> tuple[subprocess.Popen, str]:
    cmd = [sys.executable, "-m", "inferlab.fakeserver", "--first-delay", "0.02"]
    cmd += ["--interval", "0.01", "--tokens", str(tokens)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    return proc, proc.stdout.readline().strip()


def measure_once(requests: int = 150, tokens: int = 20) -> dict:
    server, url = _start_server(tokens)
    overheads_us: list[float] = []
    try:
        with httpx.Client(timeout=10) as client:
            for _ in range(5):  # warm-up: connection set-up, imports, caches
                stream_chat(client, url, {"messages": []})
            for _ in range(requests):
                result = stream_chat(client, url, {"messages": []}, keep_events=True)
                for arrived_ns, event in result.events or []:
                    if "x_sent_ns" in event:
                        overheads_us.append((arrived_ns - event["x_sent_ns"]) / 1e3)
    finally:
        server.terminate()
        server.wait(timeout=5)
    return {
        "events": len(overheads_us),
        "overhead_us": summarize(overheads_us),
        "events_that_appeared_to_arrive_early": sum(1 for v in overheads_us if v < 0),
    }


def profile_cpu(requests: int = 100, tokens: int = 20) -> tuple[str, float]:
    """Where the client's own CPU goes, using CPU time so waiting on the server is excluded."""
    server, url = _start_server(tokens)
    profile = cProfile.Profile(time.process_time)
    events = 0
    try:
        with httpx.Client(timeout=10) as client:
            for _ in range(5):
                stream_chat(client, url, {"messages": []})
            start = time.process_time()
            profile.enable()
            for _ in range(requests):
                events += len(
                    stream_chat(client, url, {"messages": []}, keep_events=True).events or []
                )
            profile.disable()
            cpu = time.process_time() - start
    finally:
        server.terminate()
        server.wait(timeout=5)
    buffer = io.StringIO()
    pstats.Stats(profile, stream=buffer).sort_stats("tottime").print_stats(12)
    return buffer.getvalue(), cpu / events * 1e6


def upper_bound(runs: Sequence[dict]) -> dict:
    """The worst value of each statistic over the runs, in microseconds."""
    keys = ("median", "p95", "p99", "max")
    return {k: max(r["overhead_us"][k] for r in runs) for k in keys}
