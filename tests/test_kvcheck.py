"""The checks that judge Day 5's predictions must pass what they should and fail what they must."""

from __future__ import annotations

import pathlib
import struct

import pytest

from inferlab import gguf, kvcheck, kvmem
from inferlab.kvmem import MIB

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "days" / "05-kv-cache-memory" / "configs"
needs_configs = pytest.mark.skipif(not CONFIGS.exists(), reason="no Day 5 configs committed")


def qwen_small() -> kvmem.Model:
    return kvmem.load_model(
        {
            "model_type": "qwen2",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "vocab_size": 100,
            "tie_word_embeddings": True,
            "use_sliding_window": False,
        }
    )


def good_summary(m: kvmem.Model, launches: int = 3) -> dict:
    """What a perfectly behaved llama.cpp would report for every configuration."""
    out = {}
    for cache_type in kvcheck.CACHE_TYPES:
        for context in kvcheck.CONTEXTS:
            cells = -(-context // 256) * 256
            mib = kvmem.kv_cache_bytes(m, cells, cache_type=cache_type) / MIB
            out[kvcheck.config_key(context, cache_type)] = {
                "context": context,
                "cache_type": cache_type,
                "cells": [cells] * launches,
                "kv_mib": [mib] * launches,
                "context_mib": [round(mib)] * launches,
                "model_mib": [100] * launches,
            }
    return out


def test_a_perfectly_behaved_engine_passes_every_kv_check():
    m = qwen_small()
    summary = good_summary(m)
    assert kvcheck.check_kv(m, summary) == []
    assert kvcheck.check_breakdown(summary, 100 * MIB) == []


def test_a_size_off_by_one_percent_fails_prediction_four():
    m = qwen_small()
    summary = good_summary(m)
    key = kvcheck.config_key(4096, "f16")
    summary[key]["kv_mib"] = [x * 1.01 for x in summary[key]["kv_mib"]]
    failures = kvcheck.check_kv(m, summary)
    assert len(failures) == 1 and key in failures[0]


def test_a_size_off_by_a_third_of_a_percent_still_passes():
    m = qwen_small()
    summary = good_summary(m)
    key = kvcheck.config_key(4096, "q8_0")
    summary[key]["kv_mib"] = [x * 1.003 for x in summary[key]["kv_mib"]]
    assert kvcheck.check_kv(m, summary) == []


def test_unpadded_cells_at_an_odd_context_fail_prediction_five():
    m = qwen_small()
    summary = good_summary(m)
    key = kvcheck.config_key(1000, "f16")
    summary[key]["cells"] = [1000] * 3
    summary[key]["kv_mib"] = [kvmem.kv_cache_bytes(m, 1000) / MIB] * 3
    failures = kvcheck.check_kv(m, summary)
    assert any("expected 1024 cells" in f for f in failures)
    assert any("above the calculator" in f for f in failures)


def test_a_missing_configuration_is_a_failure_not_a_pass():
    m = qwen_small()
    summary = good_summary(m)
    del summary[kvcheck.config_key(512, "q4_0")]
    assert kvcheck.check_kv(m, summary) == ["ctx512-q4_0: no result"]


def test_launches_that_disagree_fail_prediction_six():
    m = qwen_small()
    summary = good_summary(m)
    key = kvcheck.config_key(32768, "f16")
    summary[key]["kv_mib"] = [summary[key]["kv_mib"][0]] * 2 + [summary[key]["kv_mib"][0] + 0.01]
    assert any("differs between launches" in f for f in kvcheck.check_kv(m, summary))


def test_the_breakdown_check_catches_a_big_context_column_and_a_wrong_model_column():
    summary = good_summary(qwen_small())
    key = kvcheck.config_key(512, "f16")
    summary[key]["context_mib"] = [summary[key]["kv_mib"][0] + 3] * 3
    summary[key]["model_mib"] = [110] * 3  # 10% above 100 MiB
    failures = kvcheck.check_breakdown(summary, 100 * MIB)
    assert len(failures) == 2
    assert any("context column" in f for f in failures)
    assert any("model columns" in f for f in failures)


# ---------------------------------------------------- the GGUF-based checks


def write_gguf(tmp_path, shapes):
    def s(text):
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    records, offset, data = [], 0, b""
    for name, dims in shapes.items():
        elements = 1
        for d in dims:
            elements *= d
        records.append(
            s(name)
            + struct.pack("<I", len(dims))
            + struct.pack(f"<{len(dims)}Q", *dims)
            + struct.pack("<IQ", 0, offset)
        )
        padded = (elements * 4 + 31) // 32 * 32
        data += bytes(padded)
        offset += padded
    head = b"GGUF" + struct.pack("<IQQ", 3, len(shapes), 0) + b"".join(records)
    head += bytes((-len(head)) % 32)
    path = tmp_path / "m.gguf"
    path.write_bytes(head + data)
    return gguf.read_header(path)


def test_parameters_tensors_and_bytes_pass_for_a_file_that_matches(tmp_path):
    m = qwen_small()
    g = write_gguf(tmp_path, gguf.expected_tensors(m))
    printed = round(g.parameters / 1e6, 2)
    assert kvcheck.check_parameters(m, g, printed) == []
    assert kvcheck.check_tensors(m, g) == []
    assert kvcheck.check_weight_bytes(g) == []


def test_a_parameter_count_one_off_fails_prediction_one(tmp_path):
    m = qwen_small()
    shapes = gguf.expected_tensors(m)
    shapes["blk.0.attn_q.bias"] = (m.heads * m.head_dim + 1,)
    g = write_gguf(tmp_path, shapes)
    failures = kvcheck.check_parameters(m, g, round(g.parameters / 1e6, 2))
    assert len(failures) == 1 and "(-1)" in failures[0]


def test_an_uncaptured_or_wrong_printed_figure_fails_prediction_one(tmp_path):
    m = qwen_small()
    g = write_gguf(tmp_path, gguf.expected_tensors(m))
    assert any("not captured" in f for f in kvcheck.check_parameters(m, g, None))
    assert any("printed" in f for f in kvcheck.check_parameters(m, g, g.parameters / 1e6 + 0.1))


def test_a_data_region_that_differs_from_the_block_table_fails_prediction_three(tmp_path):
    g = write_gguf(tmp_path, gguf.expected_tensors(qwen_small()))
    g.file_size += 100 * 1024  # as if the file carried data the table does not account for
    assert kvcheck.check_weight_bytes(g)


# ---------------------------------------------------------- published figures


def test_published_figures_are_judged_by_half_a_unit_of_the_last_printed_digit():
    m = qwen_small()
    total = kvmem.param_counts(m).total / 1e9
    models = {"x": m}
    inside = {"x": {"total_b": total + 0.0004, "total_decimals": 3}}  # limit is 0.0005
    outside = {"x": {"total_b": total + 0.0006, "total_decimals": 3}}
    assert kvcheck.check_published(models, inside, active=False) == []
    assert len(kvcheck.check_published(models, outside, active=False)) == 1


def test_a_published_model_with_no_config_is_a_failure():
    out = kvcheck.check_published({}, {"y": {"total_b": 1.0, "total_decimals": 1}}, active=False)
    assert out == ["y: no model to compare with"]


def test_active_and_total_are_judged_separately():
    m = qwen_small()
    models = {"x": m}
    only_total = {"x": {"total_b": 99.0, "total_decimals": 1}}
    assert kvcheck.check_published(models, only_total, active=True) == []
    assert kvcheck.check_published(models, only_total, active=False)


@needs_configs
def test_the_latent_attention_ratio_is_checked_against_eight_to_ten():
    m = kvmem.load_config_file(CONFIGS / "DeepSeek-V2-Lite.config.json")
    assert kvcheck.check_mla(m) == []
    assert kvcheck.check_mla(qwen_small()) == ["not a latent-attention model"]
    few_heads = kvmem.Model(**(m.__dict__ | {"heads": 2}))  # expanded cache shrinks, ratio ~1.1
    assert kvcheck.check_mla(few_heads)


# ------------------------------------------------------------------ verdicts


def test_a_prediction_nobody_judged_is_reported_as_not_held():
    rows = kvcheck.verdicts({"1": []})
    assert rows[0]["held"] is True
    assert all(r["held"] is False and r["failures"] == ["not judged"] for r in rows[1:])
    assert [r["number"] for r in rows] == [str(i) for i in range(1, 11)]


def test_a_failure_makes_its_prediction_not_held_and_keeps_the_reason():
    failures = {str(i): [] for i in range(1, 11)} | {"4": ["off by 3%"]}
    rows = kvcheck.verdicts(failures)
    assert [r["number"] for r in rows if not r["held"]] == ["4"]
    assert rows[3]["failures"] == ["off by 3%"]


# ------------------------------------------------------------ launch summaries


def launch(context=512, cache_type="f16", *, ok=True, kv_lines=1, breakdown=True, flash=True):
    kv = {"size_mib": 6.0, "cells": 512, "layers": 4}
    return {
        "context": context,
        "cache_type": cache_type,
        "ok": ok,
        "error": None if ok else "exited with status 1",
        "parsed": {
            "kv": [kv] * kv_lines,
            "breakdown": {"context_mib": 6, "model_mib": 100} if breakdown else None,
            "flash_attention_enabled": flash,
        },
    }


def test_a_good_launch_has_no_problems_and_is_summarized_by_configuration():
    launches = [launch(), launch(), launch(), launch(4096, "q8_0")]
    assert all(kvcheck.launch_problems(x) == [] for x in launches)
    summary = kvcheck.summarize_launches(launches)
    assert set(summary) == {"ctx512-f16", "ctx4096-q8_0"}
    assert summary["ctx512-f16"]["kv_mib"] == [6.0, 6.0, 6.0]
    assert summary["ctx512-f16"]["context_mib"] == [6, 6, 6]


@pytest.mark.parametrize(
    "bad, fragment",
    [
        (launch(ok=False), "exited"),
        (launch(kv_lines=0), "0 KV cache lines"),
        (launch(kv_lines=2), "2 KV cache lines"),
        (launch(breakdown=False), "no memory breakdown"),
        (launch(flash=False), "flash attention"),
    ],
)
def test_an_unusable_launch_is_explained_and_left_out_of_the_summary(bad, fragment):
    assert any(fragment in p for p in kvcheck.launch_problems(bad))
    assert kvcheck.summarize_launches([bad]) == {}


def test_a_configuration_with_fewer_than_three_usable_launches_fails_the_check():
    m = qwen_small()
    summary = good_summary(m, launches=2)
    failures = kvcheck.check_kv(m, summary)
    assert failures and all("usable launches" in f for f in failures)
    assert len(failures) == len(kvcheck.CONTEXTS) * len(kvcheck.CACHE_TYPES)


def test_each_kv_prediction_is_judged_by_its_own_function():
    m = qwen_small()
    wrong_size = good_summary(m)
    key = kvcheck.config_key(4096, "f16")
    wrong_size[key]["kv_mib"] = [x * 1.1 for x in wrong_size[key]["kv_mib"]]
    assert kvcheck.check_kv_sizes(m, wrong_size) and not kvcheck.check_kv_padding(m, wrong_size)
    assert not kvcheck.check_kv_repeat(wrong_size)

    unpadded = good_summary(m)
    odd = kvcheck.config_key(1000, "q8_0")
    unpadded[odd]["cells"] = [1000] * 3
    assert kvcheck.check_kv_padding(m, unpadded) and not kvcheck.check_kv_sizes(m, unpadded)

    unstable = good_summary(m)
    unstable[key]["cells"] = [4096, 4096, 4352]
    assert kvcheck.check_kv_repeat(unstable)
    assert not kvcheck.check_kv_sizes(m, good_summary(m))


def test_a_missing_configuration_is_reported_once_not_by_every_prediction():
    m = qwen_small()
    summary = good_summary(m)
    del summary[kvcheck.config_key(1000, "f16")]
    assert kvcheck.check_kv_padding(m, summary) == ["ctx1000-f16: no result"]
    assert kvcheck.check_kv_repeat(summary) == []
