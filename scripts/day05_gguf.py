"""Read the GGUF header of the Day 5 model three times and write what it says to JSON.

Only the header is read: tensor names, shapes, quantization types and offsets, plus a few metadata
keys. The tensor data is never loaded. The summary is committed, so the checks that compare the
calculator with the file can be re-run on a machine that does not have the 469 MB model.

usage: python scripts/day05_gguf.py --out days/05-kv-cache-memory/results/gguf_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inferlab import env, gguf

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
READS = 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summaries = [gguf.to_summary(gguf.read_header(MODEL)) for _ in range(READS)]
    if any(s != summaries[0] for s in summaries[1:]):
        print("the three reads of the header disagree", file=sys.stderr)
        return 1
    summary = summaries[0] | {
        "file": str(MODEL.relative_to(ROOT)),
        "sha256": env.sha256_of(MODEL),
        "reads": READS,
        "reads_identical": True,
    }
    args.out.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {args.out}: {len(summary['tensors'])} tensors, {READS} identical reads")
    return 0


if __name__ == "__main__":
    sys.exit(main())
