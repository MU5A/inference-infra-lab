"""The numbers in a day's write-up must be the numbers in its results, not retyped copies."""

from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DAY1 = ROOT / "days" / "01-harness-and-baseline"


@pytest.mark.skipif(
    not (DAY1 / "results" / "baseline_table.md").exists(), reason="no baseline results committed"
)
def test_day_one_readme_contains_the_generated_tables_verbatim():
    generated = (DAY1 / "results" / "baseline_table.md").read_text().strip()
    readme = (DAY1 / "README.md").read_text()
    assert generated in readme, (
        "days/01-harness-and-baseline/README.md differs from results/baseline_table.md. "
        "Regenerate with: python scripts/day01_report.py"
    )


@pytest.mark.skipif(
    not (DAY1 / "results" / "followup_table.md").exists(), reason="no follow-up results committed"
)
def test_day_one_readme_contains_the_generated_follow_up_tables_verbatim():
    generated = (DAY1 / "results" / "followup_table.md").read_text().strip()
    assert generated in (DAY1 / "README.md").read_text(), (
        "days/01-harness-and-baseline/README.md differs from results/followup_table.md. "
        "Regenerate with: python scripts/day01_report.py"
    )


DAY2 = ROOT / "days" / "02-memory-hierarchy"


@pytest.mark.skipif(
    not (DAY2 / "results" / "linux_table.md").exists(), reason="no Linux results committed"
)
def test_day_two_readme_contains_the_generated_linux_tables_verbatim():
    generated = (DAY2 / "results" / "linux_table.md").read_text().strip()
    assert generated in (DAY2 / "README.md").read_text(), (
        "days/02-memory-hierarchy/README.md differs from results/linux_table.md. "
        "Regenerate with: python scripts/day02_report.py"
    )


@pytest.mark.skipif(
    not (DAY2 / "results" / "microbench_table.md").exists(), reason="no microbench results"
)
def test_day_two_readme_contains_the_generated_microbench_tables_verbatim():
    generated = (DAY2 / "results" / "microbench_table.md").read_text().strip()
    assert generated in (DAY2 / "README.md").read_text(), (
        "days/02-memory-hierarchy/README.md differs from results/microbench_table.md. "
        "Regenerate with: python scripts/day02_report.py"
    )


DAY3 = ROOT / "days" / "03-roofline"


@pytest.mark.skipif(
    not (DAY3 / "results" / "specs_table.md").exists(), reason="no Day 3 specification table"
)
def test_day_three_readme_contains_the_generated_specification_table_verbatim():
    generated = (DAY3 / "results" / "specs_table.md").read_text().strip()
    assert generated in (DAY3 / "README.md").read_text(), (
        "days/03-roofline/README.md differs from results/specs_table.md. "
        "Regenerate with: python scripts/day03_report.py"
    )


@pytest.mark.skipif(
    not (DAY3 / "results" / "measured_tables.md").exists(), reason="no Day 3 measured tables"
)
def test_day_three_readme_contains_the_generated_measured_tables_verbatim():
    generated = (DAY3 / "results" / "measured_tables.md").read_text().strip()
    assert generated in (DAY3 / "README.md").read_text(), (
        "days/03-roofline/README.md differs from results/measured_tables.md. "
        "Regenerate with: python scripts/day03_report.py"
    )


@pytest.mark.skipif(
    not (DAY3 / "results" / "analysis.json").exists(), reason="no Day 3 analysis committed"
)
def test_every_day_three_point_is_inside_the_chart_so_none_is_hidden():
    import json

    from inferlab import roofline_report as rr

    points = json.loads((DAY3 / "results" / "analysis.json").read_text())["points"]
    assert points
    for p in points:
        assert rr.X_RANGE[0] <= p["intensity"] <= rr.X_RANGE[1], p["key"]
        assert rr.Y_RANGE[0] <= p["achieved_flops"] <= rr.Y_RANGE[1], p["key"]


DAY4 = ROOT / "days" / "04-metal-programming"


@pytest.mark.skipif(
    not (DAY4 / "results" / "tables.md").exists(), reason="no Day 4 tables committed"
)
def test_day_four_readme_contains_the_generated_tables_verbatim():
    generated = (DAY4 / "results" / "tables.md").read_text().strip()
    assert generated in (DAY4 / "README.md").read_text(), (
        "days/04-metal-programming/README.md differs from results/tables.md. "
        "Regenerate with: python scripts/day04_report.py"
    )
