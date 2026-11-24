"""Render Day 4's tables from its result files, so the write-up cannot drift from them.

Reads the three phase files in days/04-metal-programming/results and two Day 3 figures (the MLX
triad median and the MLX fp32 matrix-multiply roof) straight from Day 3's result files. Writes
results/tables.md and results/analysis.json.

usage: python scripts/day04_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from inferlab import followup_report, metal_report

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "days" / "04-metal-programming" / "results"
DAY3 = ROOT / "days" / "03-roofline" / "results"
PHASES = ("stream", "occupancy", "fma")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def day3_figures() -> tuple[float, float]:
    triad = load(DAY3 / "gpu_bandwidth.json")["summary"]["triad"]
    if not (triad["stable"] and triad["validated"]):
        raise ValueError("Day 3's triad is not citable")
    return triad["bytes_per_s"]["median"], load(DAY3 / "analysis.json")["roofs"]["fp32"]


def main() -> int:
    missing = [p for p in PHASES if not (RESULTS / f"{p}.json").exists()]
    if missing:
        print(f"not all phases have results: missing {', '.join(missing)}")
        return 1
    results = {p: load(RESULTS / f"{p}.json") for p in PHASES}
    mlx_triad, mlx_fp32 = day3_figures()
    verdicts = metal_report.judge(
        results["stream"], results["occupancy"], results["fma"], mlx_triad, mlx_fp32
    )
    references = {}
    for result in results.values():
        references |= result["references"]
    mr = metal_report
    sections = [
        ("What the kernels reported about how they ran", mr.dispatch_table(results)),
        ("Streaming triad by threadgroup size", mr.stream_table(results["stream"], mlx_triad)),
        ("Threadgroup memory and bandwidth", mr.occupancy_table(results["occupancy"])),
        ("Fused multiply-add throughput by threadgroup size", mr.fma_table(results["fma"])),
        ("Predictions", mr.predictions_table(verdicts)),
        (
            "Machine speed before and after each phase",
            followup_report.references_table({"references": references}),
        ),
    ]  # fmt: skip
    text = "\n\n".join(f"#### {title}\n\n{table}" for title, table in sections) + "\n"
    (RESULTS / "tables.md").write_text(text)
    analysis = {
        "day3_mlx_triad_bytes_per_s": mlx_triad,
        "day3_mlx_fp32_matmul_flops": mlx_fp32,
        "verdicts": verdicts,
    }
    (RESULTS / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print("wrote tables.md and analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
