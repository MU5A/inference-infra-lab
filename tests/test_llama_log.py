"""Reading llama-server's JSON log: the lines it prints, and lines it should not trust."""

from __future__ import annotations

import json

from inferlab import llama_log


def log(msg: str) -> str:
    return json.dumps({"type": "log", "time": 1, "level": "info", "msg": msg + "\n"})


def breakdown(model: int, context: int, compute: int, host: int = 0) -> str:
    rows = [
        {"kind": "device", "name": "MTL0", "model": model, "context": context, "compute": compute},
        {"kind": "host", "name": "Host", "model": host, "context": 0, "compute": 0},
    ]
    return json.dumps({"type": "fit_memory_breakdown", "data": {"unit": "MiB", "rows": rows}})


KV = (
    "llama_kv_cache: size =   48.00 MiB (  4096 cells,  24 layers,  1/1 seqs), "
    "K (q8_0):   20.00 MiB, V (q8_0):   28.00 MiB"
)
SAMPLE = [
    log("print_info: file size   = 100.50 MiB (6.10 BPW) "),
    log("print_info: model params          = 630.25 M"),
    log("load_tensors:   CPU_Mapped model buffer size =    10.50 MiB"),
    log("load_tensors:  MTL0_Mapped model buffer size =    90.00 MiB"),
    log("llama_context: n_ctx                 = 4096"),
    log("llama_context:        CPU  output buffer size =     0.58 MiB"),
    log(KV),
    log("resolve_fused_ops: Flash Attention enabled"),
    log("sched_reserve:       MTL0 compute buffer size =    36.01 MiB"),
    breakdown(1, 1, 1),
    breakdown(90, 48, 36, host=10),
]


def test_the_figures_a_server_prints_are_all_read():
    r = llama_log.parse(SAMPLE)
    assert r["kv"] == [
        {
            "size_mib": 48.0,
            "cells": 4096,
            "layers": 24,
            "seqs_used": 1,
            "seqs_total": 1,
            "k_type": "q8_0",
            "k_mib": 20.0,
            "v_type": "q8_0",
            "v_mib": 28.0,
        }
    ]
    assert r["model_params_millions"] == 630.25
    assert (r["file_size_mib"], r["bits_per_weight"]) == (100.5, 6.1)
    assert r["n_ctx"] == 4096 and r["flash_attention_enabled"] is True
    assert [b["mib"] for b in r["model_buffers"]] == [10.5, 90.0]
    assert r["compute_buffers"] == [{"buffer": "MTL0", "mib": 36.01}]
    assert r["output_buffers"] == [{"buffer": "CPU", "mib": 0.58}]


def test_the_last_breakdown_is_the_one_used_and_its_columns_are_added_across_rows():
    r = llama_log.parse(SAMPLE)
    assert r["breakdown_events"] == 2
    assert (r["breakdown"]["model_mib"], r["breakdown"]["context_mib"]) == (100, 48)
    assert r["breakdown"]["compute_mib"] == 36


def test_parameter_counts_in_other_units_are_converted_to_millions():
    for unit, expected in (("K", 0.5), ("M", 500.0), ("B", 500_000.0), ("T", 500_000_000.0)):
        r = llama_log.parse([log(f"print_info: model params          = 500.00 {unit}")])
        assert r["model_params_millions"] == expected, unit
    half_a_billion = llama_log.parse([log("print_info: model params          = 0.50 B")])
    assert half_a_billion["model_params_millions"] == 500.0


def test_text_that_is_not_json_or_not_a_log_event_is_ignored_without_error():
    junk = ["", "plain text", "{not json", "[1, 2]", "{}", '{"type": "log"}', '{"msg": 5}', "null"]
    r = llama_log.parse(junk + SAMPLE)
    assert len(r["kv"]) == 1 and r["model_params_millions"] == 630.25


def test_a_run_that_printed_nothing_has_every_figure_missing_not_guessed():
    r = llama_log.parse([])
    assert r["kv"] == [] and r["model_params_millions"] is None and r["breakdown"] is None
    assert r["file_size_mib"] is None and r["n_ctx"] is None
    assert r["flash_attention_enabled"] is False and r["breakdown_events"] == 0


def test_a_server_that_died_before_shutdown_has_no_breakdown():
    r = llama_log.parse([x for x in SAMPLE if "fit_memory_breakdown" not in x])
    assert r["breakdown"] is None and len(r["kv"]) == 1


def test_two_kv_lines_are_both_kept_so_the_caller_can_refuse_the_run():
    r = llama_log.parse([log(KV), log(KV)])
    assert len(r["kv"]) == 2


def test_a_breakdown_with_odd_rows_does_not_break_the_totals():
    event = json.dumps(
        {
            "type": "fit_memory_breakdown",
            "data": {"rows": [{"model": 5, "context": "x"}, "oops", {"model": 7, "context": 2}]},
        }
    )
    r = llama_log.parse([event])
    assert r["breakdown"]["model_mib"] == 12 and r["breakdown"]["context_mib"] == 2
    bad = json.dumps({"type": "fit_memory_breakdown", "data": {"rows": "nope"}})
    assert llama_log.parse([bad])["breakdown"] is None


def test_a_cache_type_line_with_other_spacing_still_parses():
    line = (
        "llama_kv_cache: size = 1536.00 MiB ( 32768 cells,  24 layers,  1/1 seqs), "
        "K (f16):  768.00 MiB, V (f16):  768.00 MiB"
    )
    kv = llama_log.parse([log(line)])["kv"][0]
    assert (kv["size_mib"], kv["cells"], kv["k_type"], kv["v_mib"]) == (1536.0, 32768, "f16", 768.0)
