"""Render Day 3's tables and chart from its result files, so the write-up cannot drift from them.

Always writes the specification table. When all three GPU result files exist it also writes the
measured tables, the prediction verdicts, the analysis JSON and the roofline chart.

usage: python scripts/day03_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from inferlab import followup_report, roofline, roofline_report

ROOT = Path(__file__).resolve().parents[1]
DAY = ROOT / "days" / "03-roofline"
RESULTS = DAY / "results"
DAY2 = ROOT / "days" / "02-memory-hierarchy" / "results"
GPU_FILES = ("gpu_bandwidth.json", "gpu_matmul.json", "gpu_skinny.json")
GPU_COLOURS = {"fp16": "#264653", "fp32": "#e76f51"}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def write_specs() -> dict:
    specs = roofline.load_specs(DAY / "specs" / "hardware_specs.json")
    (RESULTS / "specs_table.md").write_text(roofline_report.spec_table(specs) + "\n")
    return specs


def chart_roofs(specs: dict, analysis: dict, cpu: dict) -> list[dict]:
    roofs = []
    for device in specs["devices"]:
        roof = roofline.spec_roof(device, roofline_report.DATACENTRE_COMPUTE)
        if roof:
            name = device["name"].replace("NVIDIA ", "").replace("AMD Instinct ", "")
            roofs.append({"name": name, "peak_flops": roof.peak_flops, "bandwidth": roof.bandwidth})
    r = analysis["roofs"]
    roofs.append(
        {
            "name": "This chip's CPU (Day 2)",
            "peak_flops": cpu["peak_flops"],
            "bandwidth": cpu["bandwidth"],
            "dashed": True,
            "colour": "#6c757d",
        }
    )
    for precision in ("fp32", "fp16"):
        roofs.append(
            {
                "name": f"This chip's GPU, {precision}, measured",
                "peak_flops": r[precision],
                "bandwidth": r["bandwidth"],
                "bold": True,
                "colour": GPU_COLOURS[precision],
            }
        )
    return roofs


def write_measured(specs: dict) -> None:
    bandwidth, matmul, skinny = (load(RESULTS / name) for name in GPU_FILES)
    analysis = roofline_report.analyze(bandwidth, matmul, skinny)
    cpu = roofline_report.cpu_roofs(load(DAY2 / "microbench.json"), load(DAY2 / "compute.json"))
    references = {}
    for result in (bandwidth, matmul, skinny):
        references |= result["references"]
    sections = [
        ("Roofs", roofline_report.roofs_table(analysis, cpu)),
        ("Streaming bandwidth on the GPU", roofline_report.bandwidth_table(bandwidth)),
        ("Matrix multiply on the GPU", roofline_report.matmul_table(matmul)),
        ("Decode-shaped products on the GPU", roofline_report.decode_table(analysis, skinny)),
        ("Every measured point against its roofline", roofline_report.placement_table(analysis)),
        ("Predictions", roofline_report.predictions_table(analysis)),
        (
            "Machine speed before and after each phase",
            followup_report.references_table({"references": references}),
        ),
    ]
    text = "\n\n".join(f"#### {title}\n\n{table}" for title, table in sections) + "\n"
    (RESULTS / "measured_tables.md").write_text(text)
    (RESULTS / "analysis.json").write_text(
        json.dumps(analysis | {"cpu_roofs": cpu}, indent=2) + "\n"
    )
    chart = roofline_report.roofline_svg(chart_roofs(specs, analysis, cpu), analysis["points"])
    (RESULTS / "roofline.svg").write_text(chart)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    specs = write_specs()
    if all((RESULTS / name).exists() for name in GPU_FILES):
        write_measured(specs)
        print("wrote specification table, measured tables, analysis and chart")
    else:
        print("wrote the specification table only: the GPU result files are not all present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
