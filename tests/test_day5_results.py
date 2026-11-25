"""Day 5's committed results must be internally consistent, and the analysis must be exactly what
the committed inputs produce, so a number edited by hand is caught."""

from __future__ import annotations

import json
import pathlib

import pytest

from inferlab import kvcheck, kvmem, kvreport

ROOT = pathlib.Path(__file__).resolve().parents[1]
DAY = ROOT / "days" / "05-kv-cache-memory"
RESULTS = DAY / "results"

pytestmark = pytest.mark.skipif(
    not (RESULTS / "llama_kv.json").exists(), reason="no Day 5 results committed"
)


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def inputs():
    published = load(DAY / "published.json")
    models = {
        name: kvmem.load_config_file(DAY / "configs" / f"{name}.config.json")
        for name in published["models"]
    }
    return models, load(RESULTS / "gguf_summary.json"), load(RESULTS / "llama_kv.json"), published


def test_the_committed_analysis_is_exactly_what_the_committed_inputs_produce(inputs):
    recomputed = json.loads(json.dumps(kvreport.analyze(*inputs)))
    assert recomputed == load(RESULTS / "analysis.json")


def test_the_committed_tables_are_exactly_the_rendering_of_the_analysis():
    analysis = load(RESULTS / "analysis.json")
    assert kvreport.render(analysis).strip() == (RESULTS / "tables.md").read_text().strip()


def test_every_launch_started_and_stopped_cleanly_and_none_was_set_aside(inputs):
    launches = inputs[2]["launches"]
    assert len(launches) == 36 and inputs[2]["unusable"] == []
    assert all(x["ok"] and x["exit_code"] == 0 and x["stderr_bytes"] == 0 for x in launches)
    for cache_type in kvcheck.CACHE_TYPES:
        for context in kvcheck.CONTEXTS:
            found = [
                x for x in launches if (x["context"], x["cache_type"]) == (context, cache_type)
            ]
            assert sorted(x["repeat"] for x in found) == [0, 1, 2], (context, cache_type)


def test_the_launches_used_the_model_file_that_the_header_summary_describes(inputs):
    _, gguf_summary, llama, _ = inputs
    assert llama["environment"]["model"]["sha256"] == gguf_summary["sha256"]
    assert gguf_summary["reads"] == 3 and gguf_summary["reads_identical"] is True
    assert len(gguf_summary["tensors"]) == 291


def test_the_pinned_outcome_of_each_prediction():
    """The write-up states these. If a result file changes, this fails before the prose can lie."""
    verdicts = {v["number"]: v["held"] for v in load(RESULTS / "analysis.json")["verdicts"]}
    assert verdicts == {
        "1": False,
        "2": False,
        "3": True,
        "4": True,
        "5": True,
        "6": True,
        "7": True,
        "8": False,
        "9": False,
        "10": True,
    }


def test_the_extra_tensor_is_one_embedding_matrix_and_nothing_else_differs(inputs):
    a = load(RESULTS / "analysis.json")
    p = a["parameters"]
    assert p["file_elements"] - p["calculator_tied"] == p["vocab_times_hidden"]
    assert p["calculator_output_matrix_counted"] == p["file_elements"]
    assert p["llama_printed_millions"] == round(p["file_elements"] / 1e6, 2)
    failures = {v["number"]: v["failures"] for v in a["verdicts"]}["2"]
    assert len(failures) == 1 and failures[0].startswith(
        "in the file but not expected: output.weight"
    )


def test_the_rejected_attempt_agrees_with_the_cited_run_on_every_figure():
    """The rejected README says the two runs are identical on every figure."""

    def figures(path):
        out = {}
        for x in load(path)["launches"]:
            kv, bd = x["parsed"]["kv"][0], x["parsed"]["breakdown"]
            out.setdefault((x["context"], x["cache_type"]), set()).add(
                (kv["cells"], kv["size_mib"], kv["k_mib"], kv["v_mib"], bd["model_mib"],
                 bd["context_mib"], bd["compute_mib"])
            )  # fmt: skip
        return out

    rejected = figures(RESULTS / "rejected" / "llama_kv_attempt1_parser.json")
    cited = figures(RESULTS / "llama_kv.json")
    assert rejected == cited
    assert all(len(v) == 1 for v in cited.values())


def test_the_rejected_attempt_is_marked_unusable_only_for_the_flash_attention_line():
    rejected = load(RESULTS / "rejected" / "llama_kv_attempt1_parser.json")
    assert len(rejected["unusable"]) == 36
    assert {tuple(u["problems"]) for u in rejected["unusable"]} == {
        ("flash attention was not reported as enabled",)
    }


def test_no_committed_result_file_is_large():
    for path in DAY.rglob("*"):
        if path.is_file():
            assert path.stat().st_size < 1_000_000, path
