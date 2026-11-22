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
