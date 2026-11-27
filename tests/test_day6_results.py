"""Day 6's committed results must be internally consistent, and the analysis must be exactly what
the committed inputs produce, so a number edited by hand is caught."""

from __future__ import annotations

import json
import math
import pathlib

import pytest

from inferlab import decode_model as dm
from inferlab import decode_report, moe_router

ROOT = pathlib.Path(__file__).resolve().parents[1]
DAY = ROOT / "days" / "06-prefill-decode"
RESULTS = DAY / "results"
BANDWIDTH = ROOT / "days" / "03-roofline" / "results" / "gpu_bandwidth.json"

pytestmark = pytest.mark.skipif(
    not (RESULTS / "bench.json").exists(), reason="no Day 6 results committed"
)


def load(path: pathlib.Path):
    return json.loads(path.read_text())


def same(a, b) -> bool:
    """Equal, with floats compared to a relative 1e-9: Python 3.12's sum() is compensated and
    3.11's is not, so the last bits of a recomputed float can differ between them."""
    if isinstance(a, float) or isinstance(b, float):
        return (
            isinstance(a, int | float)
            and isinstance(b, int | float)
            and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
        )
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(map(same, a, b))
    return a == b


@pytest.fixture(scope="module")
def inputs():
    summaries = {k: load(RESULTS / "gguf" / f"{k}.json") for k in dm.MODELS}
    bandwidth = load(BANDWIDTH)["summary"]["triad"]["bytes_per_s"]["median"]
    return (
        summaries,
        load(RESULTS / "bench.json"),
        load(RESULTS / "ttft.json"),
        load(RESULTS / "router.json")["scenarios"],
        bandwidth,
    )


def test_the_committed_analysis_is_exactly_what_the_committed_inputs_produce(inputs):
    recomputed = json.loads(json.dumps(decode_report.analyze(*inputs)))
    assert same(recomputed, load(RESULTS / "analysis.json"))


def test_the_float_comparison_still_catches_a_hand_edit():
    analysis = load(RESULTS / "analysis.json")
    edited = json.loads(json.dumps(analysis))
    edited["models"]["qwen2.5-0.5b-fp16"]["efficiency"] = 0.95
    assert same(analysis, json.loads(json.dumps(analysis)))
    assert not same(analysis, edited)


def test_the_committed_tables_are_exactly_the_rendering_of_the_analysis():
    analysis = load(RESULTS / "analysis.json")
    assert decode_report.render(analysis).strip() == (RESULTS / "tables.md").read_text().strip()


def test_the_committed_router_results_are_what_the_simulation_produces():
    assert same(
        load(RESULTS / "router.json")["scenarios"], json.loads(json.dumps(moe_router.scenarios()))
    )


def test_every_bench_process_ran_three_rounds_and_passed_its_probe(inputs):
    bench = inputs[1]
    for key in dm.MODELS:
        record = bench["models"][key]
        assert len(record["runs"]) == 3 and len(record["raw"]) == 3
        assert [g["round"] for g in record["gates"]] == [0, 1, 2]
        assert all(g["probe"]["drift"] == [] for g in record["gates"]), key
        for raw in record["raw"]:
            assert {e["build_commit"] for e in raw} == {"cbb1c1c"}
            assert all(len(e["samples_ts"]) == 3 and e["n_gpu_layers"] == 99 for e in raw)


def test_every_first_token_series_passed_its_probe_and_lost_no_request(inputs):
    ttft = inputs[2]
    assert set(ttft["models"]) == set(dm.MODELS)
    for key, record in ttft["models"].items():
        assert record["probe"]["drift"] == [], key
        assert all(s["failed"] == 0 for s in record["prompts"].values()), key


def test_the_header_summaries_name_the_files_that_were_measured(inputs):
    summaries = inputs[0]
    assert {k: s["file"] for k, s in summaries.items()} == dm.MODELS
    small_q4 = summaries[dm.SMALL_Q4]["sha256"]
    assert small_q4 == "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
    assert inputs[1]["environment"]["model"]["sha256"] == small_q4


def test_the_pinned_outcome_of_each_prediction():
    """The write-up states these. If a result file changes, this fails before the prose can lie."""
    verdicts = {v["number"]: v for v in load(RESULTS / "analysis.json")["verdicts"]}
    assert {n: v["held"] for n, v in verdicts.items()} == {
        "1": True,
        "2": False,
        "3": True,
        "4": True,
        "5": True,
        "6": False,
        "7": True,
        "8": True,
        "9": True,
        "10": True,
    }
    assert verdicts["2"]["failures"] == [
        "qwen2.5-0.5b-fp16: 0.96 of the bound, outside 0.55 to 0.95"
    ]
    assert verdicts["6"]["failures"] == [
        "not judged: qwen2.5-0.5b-q4_k_m tg@4096 did not repeat within 10%"
    ]


def test_the_only_unstable_bench_figure_is_the_one_the_write_up_names():
    analysis = load(RESULTS / "analysis.json")
    unstable = [
        (m, k) for m, r in analysis["models"].items() for k, s in r["rates"].items()
        if not s["stable"]
    ]  # fmt: skip
    assert unstable == [(dm.SMALL_Q4, "tg@4096")]
