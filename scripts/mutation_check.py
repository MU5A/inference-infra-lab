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
