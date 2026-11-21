"""How much does the benchmark client itself add to what it measures?

A fake server (in a separate process, as a real server would be) stamps every streamed event with
the monotonic time at which it was about to be written. The client records the time at which it
saw the same event. The difference is the harness's measurement error for that event: loopback
delivery, HTTP and line parsing, JSON decoding, and interpreter scheduling.

Three independent measurements are taken with no profiler running, each against a fresh server
and a fresh connection. A separate profiled pass, using CPU time only, shows where the client's own
CPU goes. Both processes read the same system-wide monotonic clock, which is checked by confirming
that no event ever appears to arrive before it was sent.

The timestamp is taken before the server serializes and writes each event, so server-side work is
included. The numbers are therefore an upper bound on the client's own error.
"""

import cProfile
import io
import json
import pathlib
import pstats
import subprocess
import sys
import time

import httpx

from inferlab import env
from inferlab.client import stream_chat
from inferlab.stats import summarize

OUT = pathlib.Path("/Users/musa/inference-infra-lab/days/01-harness-and-baseline/results")
RUNS = 3
REQUESTS = 150
TOKENS = 20


def start_server():
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "inferlab.fakeserver",
            "--first-delay",
            "0.02",
            "--interval",
            "0.01",
            "--tokens",
            str(TOKENS),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    return proc, proc.stdout.readline().strip()


def measure_once() -> dict:
    server, url = start_server()
    overheads_us: list[float] = []
    try:
        with httpx.Client(timeout=10) as client:
            for _ in range(5):  # warm-up: connection set-up, imports, caches
                stream_chat(client, url, {"messages": []})
            for _ in range(REQUESTS):
                result = stream_chat(client, url, {"messages": []}, keep_events=True)
                for arrived_ns, event in result.events:
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


def profile_cpu() -> tuple[str, float]:
    server, url = start_server()
    profile = cProfile.Profile(
        time.process_time
    )  # CPU time only: waiting on the server is excluded
    events = 0
    try:
        with httpx.Client(timeout=10) as client:
            for _ in range(5):
                stream_chat(client, url, {"messages": []})
            cpu_start = time.process_time()
            profile.enable()
            for _ in range(100):
                events += len(stream_chat(client, url, {"messages": []}, keep_events=True).events)
            profile.disable()
            cpu = time.process_time() - cpu_start
    finally:
        server.terminate()
        server.wait(timeout=5)
    buffer = io.StringIO()
    pstats.Stats(profile, stream=buffer).sort_stats("tottime").print_stats(12)
    return buffer.getvalue(), cpu / events * 1e6


problems = env.preflight()
if problems and "--allow-noisy" not in sys.argv:
    print("refusing to measure: " + "; ".join(problems), file=sys.stderr)
    sys.exit(3)
runs = [measure_once() for _ in range(RUNS)]
profile_text, cpu_us_per_event = profile_cpu()
(OUT / "harness_overhead_profile.txt").write_text(profile_text)
clock = time.get_clock_info("perf_counter")
report = {
    "runs": runs,
    "requests_per_run": REQUESTS,
    "client_cpu_us_per_event_profiled": cpu_us_per_event,
    "clock": {
        "implementation": clock.implementation,
        "resolution_s": clock.resolution,
        "monotonic": clock.monotonic,
    },
    "preflight_problems_overridden": problems,
    "environment": env.describe(),
    "note": "Upper bound on the harness's own error: the send timestamp precedes server "
    "serialization and the write, so server-side work is included.",
}
(OUT / "harness_overhead.json").write_text(json.dumps(report, indent=2) + "\n")
for n, r in enumerate(runs, start=1):
    s = r["overhead_us"]
    print(
        f"run {n}: events {r['events']}  overhead us  median {s['median']:.0f}  "
        f"p95 {s['p95']:.0f}  p99 {s['p99']:.0f}  max {s['max']:.0f}  "
        f"early {r['events_that_appeared_to_arrive_early']}"
    )
print(f"client CPU {cpu_us_per_event:.0f} us/event (profiled)  clock {clock.implementation}")
