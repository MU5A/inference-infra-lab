"""Command line: python -m inferlab bench --base-url http://127.0.0.1:8081 --out results.json"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import env, runner


def main() -> None:
    parser = argparse.ArgumentParser(prog="inferlab")
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("bench", help="measure TTFT, ITL and decode rate")
    bench.add_argument("--base-url", default="http://127.0.0.1:8081")
    bench.add_argument("--prompts", type=Path, required=True, help="JSON object of name to prompt")
    bench.add_argument("--runs", type=int, default=3)
    bench.add_argument("--warmup", type=int, default=1)
    bench.add_argument("--max-tokens", type=int, default=128)
    bench.add_argument("--seed", type=int, default=1)
    bench.add_argument("--warm-cache", action="store_true", help="leave the prompt cache on")
    bench.add_argument("--model", help="model file, for the recorded checksum")
    bench.add_argument("--llama-dir", help="llama.cpp checkout, for the recorded commit")
    bench.add_argument(
        "--allow-noisy",
        action="store_true",
        help="run even if the machine is busy, on battery or throttled (recorded in the results)",
    )
    bench.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    problems = env.preflight()
    if problems and not args.allow_noisy:
        print("refusing to benchmark: the machine is not in a fit state", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("fix these, or pass --allow-noisy to run anyway and record it", file=sys.stderr)
        sys.exit(3)
    prompts = json.loads(args.prompts.read_text())
    before = runner.environment(args.model, args.llama_dir)
    send, client = runner.http_sender(args.base_url)
    try:
        report = runner.run_bench(
            send,
            prompts,
            runs=args.runs,
            warmup=args.warmup,
            max_tokens=args.max_tokens,
            seed=args.seed,
            cold=not args.warm_cache,
        )
    finally:
        client.close()
    after = runner.environment(args.model, args.llama_dir)
    report["environment"] = {"before": before, "after": after, "finished_at": runner.timestamp()}
    report["preflight_problems_overridden"] = problems
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for name, block in report["results"].items():
        s = block["summary"]
        print(
            f"{name:8} prompt {s['prompt_tokens']}  ttft median {s['ttft_s']['median']}  "
            f"decode median {s['decode_tps']['median']}  failed {s['failed']}"
        )


if __name__ == "__main__":
    main()
