"""Judge Day 5's predictions and render its tables from the committed result files.

Reads the five model configurations, `published.json`, `results/gguf_summary.json` and
`results/llama_kv.json`; writes `results/analysis.json` and `results/tables.md`. The README embeds
`tables.md` verbatim and a test fails if it does not.

usage: python scripts/day05_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from inferlab import kvmem, kvreport

DAY = Path(__file__).resolve().parents[1] / "days" / "05-kv-cache-memory"


def load_inputs() -> tuple[dict, dict, dict, dict]:
    published = json.loads((DAY / "published.json").read_text())
    models = {
        name: kvmem.load_config_file(DAY / "configs" / f"{name}.config.json")
        for name in published["models"]
    }
    gguf_summary = json.loads((DAY / "results" / "gguf_summary.json").read_text())
    llama = json.loads((DAY / "results" / "llama_kv.json").read_text())
    return models, gguf_summary, llama, published


def main() -> int:
    models, gguf_summary, llama, published = load_inputs()
    analysis = kvreport.analyze(models, gguf_summary, llama, published)
    (DAY / "results" / "analysis.json").write_text(json.dumps(analysis, indent=1) + "\n")
    (DAY / "results" / "tables.md").write_text(kvreport.render(analysis) + "\n")
    held = sum(1 for v in analysis["verdicts"] if v["held"])
    print(f"{held} of {len(analysis['verdicts'])} predictions held")
    for v in analysis["verdicts"]:
        print(f"  {v['number']:>2} {'held' if v['held'] else 'FAILED'}  {v['prediction']}")
        for f in [] if v["held"] else v["failures"]:
            print(f"        {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
