"""Check that the tests can fail: break the code on purpose, one plausible mistake at a time.

Each mutation edits one source file, runs the test suite, and requires it to FAIL. A mutation that
the suite does not catch is a gap in the tests. Files are restored after each mutation, and the
script refuses to start if the source tree has uncommitted changes, so `git checkout -- src`
always recovers the original if a run is interrupted.

Usage:  python scripts/mutation_check.py [substring of a mutation name ...]
Exit status is 1 if any mutation was missed.
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
S = ROOT / "src" / "inferlab"
C = ROOT / "microbench"

MUTATIONS = [
    (
        "percentile: no interpolation",
        S / "stats.py",
        "return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))",
        "return float(ordered[low])",
    ),
    (
        "client: any event counts as text (TTFT too early)",
        S / "client.py",
        "                if text:\n                    result.text_arrivals_ns.append(arrived)",
        "                if True:\n                    result.text_arrivals_ns.append(arrived)",
    ),
    (
        "decode rate off by one",
        S / "client.py",
        "decode_tps = (len(arrivals) - 1) / span",
        "decode_tps = len(arrivals) / span",
    ),
    (
        "usage rate off by one",
        S / "client.py",
        "decode_tps_usage = (result.n_tokens - 1) / span",
        "decode_tps_usage = result.n_tokens / span",
    ),
    (
        "network errors raise instead of being returned",
        S / "client.py",
        "    except httpx.HTTPError as exc:",
        "    except ZeroDivisionError as exc:",
    ),
    (
        "[DONE] treated as malformed",
        S / "client.py",
        '    if payload == "[DONE]":\n        return "done", None\n',
        "",
    ),
    (
        "ITL gaps wrong pairing",
        S / "client.py",
        "gaps = [(b - a) / 1e9 for a, b in zip(arrivals, arrivals[1:], strict=False)]",
        "gaps = [(b - a) / 1e9 for a, b in zip(arrivals, arrivals[2:], strict=False)]",
    ),
    (
        "warm-up requests are measured too",
        S / "runner.py",
        "measured = [metrics(send(payload)) for _ in range(runs)]",
        "measured = [metrics(send(payload)) for _ in range(runs + warmup)]",
    ),
    (
        "cold flag ignored",
        S / "runner.py",
        '    if cold:\n        payload["cache_prompt"] = False',
        '    if False:\n        payload["cache_prompt"] = False',
    ),
    ("failures not counted", S / "runner.py", '"failed": len(runs) - len(ok),', '"failed": 0,'),
    (
        "ITL pooled from first run only",
        S / "runner.py",
        'pooled_itl = [gap for r in ok for gap in r["itl_s"]]',
        'pooled_itl = list(ok[0]["itl_s"]) if ok else []',
    ),
    ("checksum sidecar ignored", S / "env.py", "        if sidecar.exists():", "        if False:"),
    (
        "preflight: load threshold ignored",
        S / "env.py",
        "if load is not None and load > cores * max_load_per_core:",
        "if False:",
    ),
    (
        "preflight: battery never refused",
        S / "env.py",
        'if require_ac and power and "Battery Power" in power:',
        "if False:",
    ),
    (
        "preflight: throttle limit comparison reversed",
        S / "env.py",
        "if int(value) < 100:",
        "if int(value) > 100:",
    ),
    (
        "preflight: override not recorded (CLI)",
        S / "__main__.py",
        "if problems and not args.allow_noisy:",
        "if False:",
    ),
    (
        "gpu: a busy GPU is never refused",
        S / "env.py",
        'if gpu and (gpu["peak"] > max_gpu_peak or gpu["mean"] > max_gpu_mean):',
        "if False:",
    ),
    (
        "gpu: the mean limit is ignored",
        S / "env.py",
        ' or gpu["mean"] > max_gpu_mean',
        "",
    ),
    (
        "stability: the decode limit is disabled",
        S / "runner.py",
        "MAX_DECODE_CV = 0.05",
        "MAX_DECODE_CV = 1e9",
    ),
    (
        "stability: the stable flag is always true",
        S / "runner.py",
        '        "stable": stable,',
        '        "stable": True,',
    ),
    (
        "cli: unstable results exit successfully",
        S / "__main__.py",
        'if not report["stable"]:',
        "if False:",
    ),
    (
        "stats: sample instead of population deviation",
        S / "stats.py",
        "variance = sum((v - mean) ** 2 for v in data) / len(data)",
        "variance = sum((v - mean) ** 2 for v in data) / (len(data) - 1)",
    ),
    (
        "chase: Fisher-Yates instead of Sattolo (short cycles allowed)",
        C / "chase.c",
        "rng_next(&state) % i);",
        "rng_next(&state) % (i + 1));",
    ),
    (
        "chase: every node points at itself",
        C / "chase.c",
        "base + (size_t)next[i] * node_bytes;",
        "base + i * node_bytes;",
    ),
    (
        "membw: the triad kernel computes the wrong thing",
        C / "membw.c",
        "a[i] = b[i] + SCALAR * c[i];",
        "a[i] = b[i] + c[i];",
    ),
    (
        "membw: the scale kernel does nothing",
        C / "membw.c",
        "for (size_t i = j->lo; i < j->hi; i++) b[i] = SCALAR * c[i];",
        "(void)b;",
    ),
    (
        "matmul: ikj forgets to zero C between calls",
        C / "matmul.c",
        "static void mm_ikj(int n, const float *A, const float *B, float *C) {\n    memset(C, 0, (size_t)n * n * sizeof(float));",
        "static void mm_ikj(int n, const float *A, const float *B, float *C) {",
    ),
    (
        "matmul: blocked tiles run past the edge",
        C / "matmul.c",
        "jmax = jj + bs < n ? jj + bs : n;",
        "jmax = jj + bs;",
    ),
    (
        "mmap_touch: the stride is off by one",
        C / "mmap_touch.c",
        "off += stride) checksum",
        "off += stride + 1) checksum",
    ),
    (
        "copy_nocache: an existing destination is appended to",
        C / "copy_nocache.c",
        "O_WRONLY | O_CREAT | O_TRUNC",
        "O_WRONLY | O_CREAT | O_APPEND",
    ),
    (
        "memory: plateau tolerance is far too tight",
        S / "memory.py",
        "PLATEAU_TOLERANCE = 0.30",
        "PLATEAU_TOLERANCE = 0.01",
    ),
    (
        "memory: two points are enough for a plateau",
        S / "memory.py",
        "MIN_PLATEAU_POINTS = 3",
        "MIN_PLATEAU_POINTS = 2",
    ),
    (
        "memory: the bandwidth stability limit is loosened",
        S / "memory.py",
        '"bandwidth": 0.10',
        '"bandwidth": 0.90',
    ),
    (
        "memory: low free memory is never refused",
        S / "env.py",
        "if free is not None and free < min_free_memory_percent:",
        "if False:",
    ),
    (
        "memory: the free-percentage regex reads the wrong number",
        S / "env.py",
        'r"free percentage:\\s*(\\d+)%"',
        'r"(\\d+)%"',
    ),
    (
        "env: busiest processes sorted the wrong way round",
        S / "env.py",
        "rows.sort(key=lambda r: r[2], reverse=True)",
        "rows.sort(key=lambda r: r[2])",
    ),
    (
        "env: the benchmark's own process tree is not excluded",
        S / "env.py",
        "rows = [r for r in rows if r[0] not in excluded]",
        "rows = list(rows)",
    ),
    (
        "env: executable path is kept instead of just the name",
        S / "env.py",
        'return [{"cpu_percent": r[2], "name": Path(r[3]).name} for r in rows[:n]]',
        'return [{"cpu_percent": r[2], "name": r[3]} for r in rows[:n]]',
    ),
    (
        "memory: a cold read need only match a cold touch",
        S / "memory.py",
        'if trial["cold_read"]["gbs"] < 5 * plain:',
        'if trial["cold_read"]["gbs"] < 0 * plain:',
    ),
    (
        "memory: the willneed prediction is never checked",
        S / "memory.py",
        'if trial["cold_willneed"]["gbs"] < 3 * plain:',
        "if False:",
    ),
    (
        "matmul: every sample times a single call, however short",
        C / "matmul.c",
        "int calls = probe_ns >= MIN_SAMPLE_NS ? 1 :",
        "int calls = probe_ns >= 0 ? 1 :",
    ),
    (
        "crosscheck: the cache speed-up prediction is never checked",
        S / "crosscheck.py",
        'if row["speedup"] < min_speedup:',
        "if False:",
    ),
    (
        "crosscheck: a cold run that reused cached tokens passes",
        S / "crosscheck.py",
        'if row["cold_cached_tokens"] != 0:',
        "if False:",
    ),
    (
        "crosscheck: llama-bench variation is not limited",
        S / "crosscheck.py",
        "MAX_BENCH_CV = 0.05",
        "MAX_BENCH_CV = 9.0",
    ),
    (
        "crosscheck: the decode tolerance is loosened",
        S / "crosscheck.py",
        "DECODE_TOLERANCE = 0.10",
        "DECODE_TOLERANCE = 0.90",
    ),
    (
        "overhead: the upper bound takes the best run, not the worst",
        S / "overhead.py",
        'return {k: max(r["overhead_us"][k] for r in runs) for k in keys}',
        'return {k: min(r["overhead_us"][k] for r in runs) for k in keys}',
    ),
    (
        "watch: contended is never reported",
        S / "watch.py",
        '"contended": peak is not None and peak > CONTENDED_CPU_PERCENT,',
        '"contended": False,',
    ),
    (
        "watch: wait_quiet never waits",
        S / "watch.py",
        "if not problems or clock() >= deadline:",
        "if True:",
    ),
    (
        "usage not captured",
        S / "client.py",
        '                if isinstance(event.get("usage"), dict):\n                    result.usage = event["usage"]\n',
        "",
    ),
]


def main() -> int:
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--", "src"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        print("refusing to run: src/ has uncommitted changes\n" + dirty, file=sys.stderr)
        return 2
    only = sys.argv[1:]
    missed = 0
    for name, path, old, new in MUTATIONS:
        if only and not any(o in name for o in only):
            continue
        original = path.read_text()
        if old not in original:
            print(f"SKIP (pattern not found) {name}")
            missed += 1
            continue
        path.write_text(original.replace(old, new, 1))
        try:
            run = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
        finally:
            path.write_text(original)
        caught = run.returncode != 0
        failed = next((ln for ln in run.stdout.splitlines() if ln.startswith("FAILED")), "")
        print(f"{'CAUGHT ' if caught else 'MISSED!'} {name}")
        if caught and failed:
            print(f"          {failed[:110]}")
        missed += not caught
    print(f"missed or skipped: {missed}")
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
