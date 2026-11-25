"""Day 5: the checks that judge the ten predictions written in `kvmem.py`, each returning a list of
plain-language failures (empty means the prediction held). The thresholds are fixed here, before
any measurement, and are not changed after a result is seen.

Input shapes
------------
`summary` maps a configuration key (`config_key(context, cache_type)`) to what llama.cpp reported
across the independent launches of that configuration:

    {"context": 512, "cache_type": "f16",
     "cells": [512, 512, 512],            cells the KV line reported, one per launch
     "kv_mib": [6.0, 6.0, 6.0],           total KV size on that line
     "context_mib": [7, 7, 7],            the `context` column of the shutdown breakdown
     "model_mib": [469, 469, 469]}        the `model` column added across the breakdown rows

`published` maps a model name to the figures its model card or paper prints, in billions, with the
number of decimals printed (the tolerance is half a unit of the last digit):

    {"Mixtral-8x7B-v0.1": {"total_b": 46.7, "total_decimals": 1,
                           "active_b": 12.9, "active_decimals": 1}}
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence

from . import kvmem
from .gguf import GGUF, compare_tensors, expected_tensors
from .kvmem import MIB, Model

CONTEXTS = (512, 1000, 4096, 32768)
CACHE_TYPES = ("f16", "q8_0", "q4_0")
PAD_MULTIPLE = 256
LAUNCHES_REQUIRED = 3

# Fixed before any measurement.
PARAMS_TOLERANCE_MILLIONS = 0.005  # llama.cpp prints two decimals of millions
WEIGHT_BYTES_TOLERANCE = 1e-4
KV_TOLERANCE = 0.005
PADDING_MIN_OVER = 0.02
CONTEXT_COLUMN_MIB = 2
MODEL_COLUMN_TOLERANCE = 0.02
MLA_RATIO_RANGE = (8.0, 10.0)


def config_key(context: int, cache_type: str) -> str:
    return f"ctx{context}-{cache_type}"


def _close(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance * abs(b)


def check_parameters(m: Model, g: GGUF, printed_millions: float | None) -> list[str]:
    """Prediction 1."""
    failures = []
    calculated = kvmem.param_counts(m).total
    if calculated != g.parameters:
        failures.append(
            f"calculator {calculated} parameters, the file's tensors hold {g.parameters} "
            f"({calculated - g.parameters:+d})"
        )
    if printed_millions is None:
        failures.append("llama.cpp's printed `model params` was not captured")
    elif abs(g.parameters / 1e6 - printed_millions) > PARAMS_TOLERANCE_MILLIONS + 1e-9:
        failures.append(
            f"llama.cpp printed {printed_millions} M, the file's tensors hold "
            f"{g.parameters / 1e6:.4f} M"
        )
    return failures


def check_tensors(m: Model, g: GGUF) -> list[str]:
    """Prediction 2."""
    return compare_tensors(expected_tensors(m), g)


def check_weight_bytes(g: GGUF) -> list[str]:
    """Prediction 3."""
    computed, actual = g.padded_bytes(), g.data_bytes
    if not _close(computed, actual, WEIGHT_BYTES_TOLERANCE):
        return [
            f"block table gives {computed} bytes, the file's tensor-data region is {actual} "
            f"({computed - actual:+d})"
        ]
    return []


def _median(summary: Mapping, key: str, field: str) -> float:
    return statistics.median(summary[key][field])


def _configurations(contexts: Sequence[int]):
    for cache_type in CACHE_TYPES:
        for context in contexts:
            yield context, cache_type, config_key(context, cache_type)


def check_kv_sizes(m: Model, summary: Mapping[str, Mapping]) -> list[str]:
    """Prediction 4: contexts that are multiples of 256."""
    failures = []
    for context, cache_type, key in _configurations([c for c in CONTEXTS if c % PAD_MULTIPLE == 0]):
        if key not in summary:
            failures.append(f"{key}: no result")
            continue
        cells = _median(summary, key, "cells")
        reported = _median(summary, key, "kv_mib")
        want = kvmem.kv_cache_bytes(m, context, cache_type=cache_type) / MIB
        if cells != context or not _close(reported, want, KV_TOLERANCE):
            failures.append(
                f"{key}: llama.cpp reported {reported} MiB over {cells:.0f} cells, "
                f"calculator {want:.2f} MiB for {context}"
            )
    return failures


def check_kv_padding(m: Model, summary: Mapping[str, Mapping]) -> list[str]:
    """Prediction 5: a context that is not a multiple of 256."""
    failures = []
    for context, cache_type, key in _configurations([c for c in CONTEXTS if c % PAD_MULTIPLE != 0]):
        if key not in summary:
            failures.append(f"{key}: no result")
            continue
        cells = _median(summary, key, "cells")
        reported = _median(summary, key, "kv_mib")
        padded = -(-context // PAD_MULTIPLE) * PAD_MULTIPLE
        at_padded = kvmem.kv_cache_bytes(m, padded, cache_type=cache_type) / MIB
        at_asked = kvmem.kv_cache_bytes(m, context, cache_type=cache_type) / MIB
        if cells != padded:
            failures.append(f"{key}: expected {padded} cells, reported {cells:.0f}")
        if not _close(reported, at_padded, KV_TOLERANCE):
            failures.append(
                f"{key}: reported {reported} MiB, calculator {at_padded:.2f} MiB at {padded} cells"
            )
        if reported < (1 + PADDING_MIN_OVER) * at_asked:
            failures.append(
                f"{key}: reported {reported} MiB is not {PADDING_MIN_OVER:.0%} above the "
                f"calculator's {at_asked:.2f} MiB for {context} tokens"
            )
    return failures


def check_kv_repeat(summary: Mapping[str, Mapping]) -> list[str]:
    """Prediction 6: three usable launches of every configuration, all reporting the same."""
    failures = []
    for _, _, key in _configurations(CONTEXTS):
        if key not in summary:
            continue  # a missing configuration is reported by predictions 4 and 5
        row = summary[key]
        if len(row["kv_mib"]) < LAUNCHES_REQUIRED:
            failures.append(
                f"{key}: {len(row['kv_mib'])} usable launches, {LAUNCHES_REQUIRED} required"
            )
        for field in ("cells", "kv_mib"):
            if len(set(row[field])) > 1:
                failures.append(f"{key}: {field} differs between launches: {row[field]}")
    return failures


def check_kv(m: Model, summary: Mapping[str, Mapping]) -> list[str]:
    """Predictions 4, 5 and 6 together."""
    return check_kv_sizes(m, summary) + check_kv_padding(m, summary) + check_kv_repeat(summary)


def check_breakdown(summary: Mapping[str, Mapping], region_bytes: int) -> list[str]:
    """Prediction 7."""
    failures = []
    region_mib = region_bytes / MIB
    for key, row in summary.items():
        columns = zip(row["kv_mib"], row["context_mib"], row["model_mib"], strict=True)
        for kv, context, model in columns:
            if abs(context - kv) > CONTEXT_COLUMN_MIB:
                failures.append(f"{key}: context column {context} MiB against KV {kv} MiB")
            if not _close(model, region_mib, MODEL_COLUMN_TOLERANCE):
                failures.append(
                    f"{key}: model columns add to {model} MiB, tensor data is {region_mib:.1f} MiB"
                )
    return sorted(set(failures))


def _within_rounding(calculated: float, printed: float, decimals: int) -> bool:
    return abs(calculated - printed) <= 0.5 * 10**-decimals + 1e-9


def check_published(
    models: Mapping[str, Model], published: Mapping[str, Mapping], *, active: bool
) -> list[str]:
    """Prediction 8 (totals) or 9 (activated parameters)."""
    failures = []
    field = "active" if active else "total"
    for name, figures in published.items():
        if f"{field}_b" not in figures:
            continue
        if name not in models:
            failures.append(f"{name}: no model to compare with")
            continue
        counts = kvmem.param_counts(models[name])
        calculated = (counts.active if active else counts.total) / 1e9
        printed = figures[f"{field}_b"]
        if not _within_rounding(calculated, printed, figures[f"{field}_decimals"]):
            failures.append(
                f"{name}: calculator {calculated:.4f} B {field}, published {printed} B "
                f"({calculated - printed:+.4f})"
            )
    return failures


def check_mla(m: Model) -> list[str]:
    """Prediction 10."""
    if not m.mla:
        return ["not a latent-attention model"]
    compressed = kvmem.kv_bytes_per_token(m, "f16", mla="compressed")
    expanded = kvmem.kv_bytes_per_token(m, "f16", mla="expanded")
    ratio = expanded / compressed
    low, high = MLA_RATIO_RANGE
    if not low <= ratio <= high:
        return [f"expanded cache is {ratio:.2f} times the compressed one, outside {low} to {high}"]
    return []


def launch_problems(launch: Mapping) -> list[str]:
    """Why a launch cannot be used. Empty means it can. An unusable launch is kept and reported."""
    problems = []
    if not launch.get("ok"):
        problems.append(launch.get("error") or "the server did not start or stop cleanly")
        return problems
    parsed = launch["parsed"]
    if len(parsed["kv"]) != 1:
        problems.append(f"{len(parsed['kv'])} KV cache lines, expected exactly 1")
    if parsed["breakdown"] is None:
        problems.append("no memory breakdown at shutdown")
    if not parsed["flash_attention_enabled"]:
        problems.append("flash attention was not reported as enabled")
    return problems


def summarize_launches(launches: Sequence[Mapping]) -> dict[str, dict]:
    """Group usable launches by configuration into the shape the checks take."""
    out: dict[str, dict] = {}
    for launch in launches:
        if launch_problems(launch):
            continue
        kv = launch["parsed"]["kv"][0]
        breakdown = launch["parsed"]["breakdown"]
        row = out.setdefault(
            config_key(launch["context"], launch["cache_type"]),
            {
                "context": launch["context"],
                "cache_type": launch["cache_type"],
                "cells": [],
                "kv_mib": [],
                "context_mib": [],
                "model_mib": [],
            },
        )
        row["cells"].append(kv["cells"])
        row["kv_mib"].append(kv["size_mib"])
        row["context_mib"].append(breakdown["context_mib"])
        row["model_mib"].append(breakdown["model_mib"])
    return out


PREDICTIONS = (
    ("1", "Parameter count equals the GGUF's tensor elements exactly and llama.cpp's printed one"),
    ("2", "The calculator's tensors are exactly the GGUF's tensors, names and shapes"),
    ("3", "Block-table tensor sizes add to the file's tensor-data region within 0.01%"),
    ("4", "KV size within 0.5% of llama.cpp at 512, 4096 and 32768 for f16, q8_0, q4_0"),
    ("5", "1000 tokens of context are reported as 1024 cells, 2% or more above the 1000 figure"),
    ("6", "The reported KV size is identical across the three launches of each configuration"),
    ("7", "Breakdown context column within 2 MiB of KV; model columns within 2% of tensor data"),
    ("8", "Total parameters of all five models within the rounding of the published figure"),
    ("9", "Activated parameters of the three mixture-of-experts models within published rounding"),
    ("10", "DeepSeek-V2-Lite's compressed cache is 8 to 10 times smaller than the expanded one"),
)  # fmt: skip


def verdicts(failures: Mapping[str, Sequence[str]]) -> list[dict]:
    """One row per prediction. A prediction nobody judged is reported as not held."""
    rows = []
    for number, text in PREDICTIONS:
        found = list(failures[number]) if number in failures else ["not judged"]
        rows.append({"number": number, "prediction": text, "held": not found, "failures": found})
    return rows
