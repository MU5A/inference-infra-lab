"""Judge Day 6's predictions and render its tables from the committed result files.

Reads `results/gguf/*.json`, `results/bench.json`, `results/ttft.json`, `results/router.json` and
Day 3's bandwidth result; writes `results/analysis.json` and `results/tables.md`.

usage: python scripts/day06_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from inferlab import decode_model as dm
from inferlab import decode_report

DAYS = Path(__file__).resolve().parents[1] / "days"
DAY = DAYS / "06-prefill-decode"
BANDWIDTH_FILE = DAYS / "03-roofline" / "results" / "gpu_bandwidth.json"


def load_inputs() -> tuple[dict, dict, dict, list, float]:
    results = DAY / "results"

    def read(path: Path):
        return json.loads(path.read_text())

    summaries = {k: read(results / "gguf" / f"{k}.json") for k in dm.MODELS}
    bandwidth = read(BANDWIDTH_FILE)["summary"]["triad"]["bytes_per_s"]["median"]
    return (
        summaries,
        read(results / "bench.json"),
        read(results / "ttft.json"),
        read(results / "router.json")["scenarios"],
        bandwidth,
    )


def main() -> int:
    analysis = decode_report.analyze(*load_inputs())
    (DAY / "results" / "analysis.json").write_text(json.dumps(analysis, indent=1) + "\n")
    (DAY / "results" / "tables.md").write_text(decode_report.render(analysis) + "\n")
    held = sum(1 for v in analysis["verdicts"] if v["held"])
    print(f"{held} of {len(analysis['verdicts'])} predictions held")
    for v in analysis["verdicts"]:
        print(f"  {v['number']} {'held' if v['held'] else 'FAILED'}  {v['prediction']}")
        for f in [] if v["held"] else v["failures"]:
            print(f"      {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
