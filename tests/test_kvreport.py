"""The report must re-judge the predictions from the result files alone, and say what it finds."""

from __future__ import annotations

import json
import pathlib

import pytest

from inferlab import gguf, kvcheck, kvmem, kvreport
from inferlab.kvmem import MIB

ROOT = pathlib.Path(__file__).resolve().parents[1]
DAY = ROOT / "days" / "05-kv-cache-memory"
needs_configs = pytest.mark.skipif(not DAY.exists(), reason="no Day 5 files committed")


def load_models() -> tuple[dict, dict]:
    published = json.loads((DAY / "published.json").read_text())
    models = {
        name: kvmem.load_config_file(DAY / "configs" / f"{name}.config.json")
        for name in published["models"]
    }
    return models, published


def synthetic_gguf_summary(m: kvmem.Model, *, output_matrix: bool) -> dict:
    """A summary of a file laid out as the calculator expects, optionally with a separate output
    matrix as the real file has. Every tensor is float32 so the sizes are easy to know."""
    shapes = dict(gguf.expected_tensors(m))
    if output_matrix and "output.weight" not in shapes:
        shapes["output.weight"] = (m.hidden, m.vocab)
    tensors, offset = [], 0
    for name, dims in shapes.items():
        elements = 1
        for d in dims:
            elements *= d
        tensors.append({"name": name, "dims": list(dims), "type_id": 0, "offset": offset})
        offset += (elements * 4 + 31) // 32 * 32
    return {
        "version": 3,
        "alignment": 32,
        "data_offset": 1024,
        "file_size": 1024 + offset,
        "metadata": {},
        "tensors": tensors,
    }


def perfect_launches(m: kvmem.Model, printed_millions: float) -> list[dict]:
    """What an engine that agreed with the calculator would log, three launches of each setting."""
    launches = []
    for repeat in range(3):
        for cache_type in kvcheck.CACHE_TYPES:
            for context in kvcheck.CONTEXTS:
                cells = -(-context // 256) * 256
                mib = kvmem.kv_cache_bytes(m, cells, cache_type=cache_type) / MIB
                launches.append(
                    {
                        "context": context,
                        "cache_type": cache_type,
                        "repeat": repeat,
                        "ok": True,
                        "error": None,
                        "parsed": {
                            "kv": [
                                {
                                    "size_mib": mib,
                                    "cells": cells,
                                    "k_mib": mib / 2,
                                    "v_mib": mib / 2,
                                }
                            ],
                            "breakdown": {
                                "context_mib": round(mib),
                                "model_mib": 100,
                                "compute_mib": 30,
                            },
                            "flash_attention_enabled": True,
                            "model_params_millions": printed_millions,
                        },
                    }
                )
    return launches


@needs_configs
def test_with_an_engine_that_agrees_with_the_calculator_the_file_and_kv_predictions_hold():
    models, published = load_models()
    qwen = models[kvreport.QWEN]
    summary = synthetic_gguf_summary(qwen, output_matrix=False)
    g = gguf.from_summary(summary)
    launches = perfect_launches(qwen, round(g.parameters / 1e6, 2))
    a = kvreport.analyze(models, summary, {"launches": launches}, published)
    held = {v["number"]: v["held"] for v in a["verdicts"]}
    assert held["1"] and held["2"] and held["3"] and held["4"] and held["5"] and held["6"]
    assert held["10"]
    # the published-figure predictions do not depend on the engine: these two fail on real data
    assert held["8"] is False and held["9"] is False


@needs_configs
def test_a_file_with_a_separate_output_matrix_fails_predictions_one_and_two_by_that_matrix():
    models, published = load_models()
    qwen = models[kvreport.QWEN]
    summary = synthetic_gguf_summary(qwen, output_matrix=True)
    a = kvreport.analyze(models, summary, {"launches": perfect_launches(qwen, 1.0)}, published)
    by_number = {v["number"]: v for v in a["verdicts"]}
    assert not by_number["1"]["held"] and not by_number["2"]["held"]
    assert "output.weight" in by_number["2"]["failures"][0]
    assert (
        a["parameters"]["file_elements"] - a["parameters"]["calculator_tied"]
        == (a["parameters"]["vocab_times_hidden"])
    )
    assert a["parameters"]["calculator_output_matrix_counted"] == a["parameters"]["file_elements"]


@needs_configs
def test_a_missing_configuration_or_an_unusable_launch_cannot_pass_quietly():
    models, published = load_models()
    qwen = models[kvreport.QWEN]
    summary = synthetic_gguf_summary(qwen, output_matrix=False)
    launches = perfect_launches(qwen, 1.0)
    launches = [x for x in launches if not (x["context"] == 512 and x["cache_type"] == "q4_0")]
    launches[0] = launches[0] | {"ok": False, "error": "the server exited with status 1"}
    a = kvreport.analyze(models, summary, {"launches": launches}, published)
    by_number = {v["number"]: v for v in a["verdicts"]}
    assert not by_number["4"]["held"]
    assert not by_number["6"]["held"]
    assert {u["config"] for u in a["unusable_launches"]} == {
        kvcheck.config_key(launches[0]["context"], launches[0]["cache_type"])
    }


@needs_configs
def test_the_tables_name_every_configuration_model_and_prediction():
    models, published = load_models()
    qwen = models[kvreport.QWEN]
    summary = synthetic_gguf_summary(qwen, output_matrix=True)
    a = kvreport.analyze(models, summary, {"launches": perfect_launches(qwen, 1.0)}, published)
    text = kvreport.render(a)
    for name in published["models"]:
        assert name in text
    assert text.count("\n| 512 | f16 |") == 2  # the KV table and the breakdown table
    assert "(compressed cache)" in text and "(expanded cache)" in text
    assert "added after prediction 1 failed" in text
    for number in range(1, 11):
        assert f"\n| {number} | " in text


def test_a_markdown_table_has_one_separator_and_one_line_per_row():
    table = kvreport.md(["a", "b"], [(1, 2), (3, 4)])
    assert table.splitlines() == ["| a | b |", "|---|---|", "| 1 | 2 |", "| 3 | 4 |"]


def test_the_printed_parameter_count_is_used_only_if_every_launch_agrees():
    ok = {"ok": True, "parsed": {"model_params_millions": 630.17}}
    other = {"ok": True, "parsed": {"model_params_millions": 494.03}}
    failed = {"ok": False, "parsed": {"model_params_millions": 1.0}}
    assert kvreport.printed_parameters([ok, ok, failed]) == 630.17
    assert kvreport.printed_parameters([ok, other]) is None
    assert kvreport.printed_parameters([]) is None
