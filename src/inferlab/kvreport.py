"""Day 5: judge the predictions from the committed result files and render the tables.

Everything here is a pure function of four committed inputs: the model configurations, the GGUF
header summary (`results/gguf_summary.json`), the llama.cpp launches (`results/llama_kv.json`) and
the published figures (`published.json`). The model file itself is not needed, so a test can
recompute the verdicts and the tables on any machine, and a README that retypes a number differs
from them and fails.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence

from . import kvcheck, kvmem
from .gguf import from_summary
from .kvmem import GIB, MIB, Model

SCENARIO_CONTEXTS = (4096, 32768, 131072)
SCENARIO_BITS = 16.0
QWEN = "Qwen2.5-0.5B-Instruct"
DEEPSEEK = "DeepSeek-V2-Lite"


def md(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _med(values: Sequence[float]) -> float:
    return statistics.median(values)


def printed_parameters(launches: Sequence[Mapping]) -> float | None:
    """The `model params` figure llama.cpp printed, if every usable launch printed the same."""
    seen = {
        x["parsed"]["model_params_millions"]
        for x in launches
        if x.get("ok") and x["parsed"]["model_params_millions"] is not None
    }
    return seen.pop() if len(seen) == 1 else None


def analyze(
    models: Mapping[str, Model],
    gguf_summary: Mapping,
    llama: Mapping,
    published: Mapping,
) -> dict:
    qwen = models[QWEN]
    g = from_summary(gguf_summary)
    launches = llama["launches"]
    summary = kvcheck.summarize_launches(launches)
    printed = printed_parameters(launches)
    pub = published["models"]
    failures = {
        "1": kvcheck.check_parameters(qwen, g, printed),
        "2": kvcheck.check_tensors(qwen, g),
        "3": kvcheck.check_weight_bytes(g),
        "4": kvcheck.check_kv_sizes(qwen, summary),
        "5": kvcheck.check_kv_padding(qwen, summary),
        "6": kvcheck.check_kv_repeat(summary),
        "7": kvcheck.check_breakdown(summary, g.data_bytes),
        "8": kvcheck.check_published(models, pub, active=False),
        "9": kvcheck.check_published(models, pub, active=True),
        "10": kvcheck.check_mla(models[DEEPSEEK]),
    }
    untied = kvmem.Model(**(qwen.__dict__ | {"tied": False}))
    by_type: dict[str, dict] = {}
    for t in g.tensors:
        row = by_type.setdefault(t.type_name, {"tensors": 0, "elements": 0, "bytes": 0})
        row["tensors"] += 1
        row["elements"] += t.elements
        row["bytes"] += t.nbytes
    weights = [t for t in g.tensors if len(t.dims) == 2]
    k_quant = {"Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K"}
    divisible = [t for t in weights if t.dims[0] % 256 == 0]
    return {
        "verdicts": kvcheck.verdicts(failures),
        "parameters": {
            "calculator_tied": kvmem.param_counts(qwen).total,
            "calculator_output_matrix_counted": kvmem.param_counts(untied).total,
            "file_elements": g.parameters,
            "llama_printed_millions": printed,
            "vocab_times_hidden": qwen.vocab * qwen.hidden,
        },
        "file": {
            "tensors": len(g.tensors),
            "data_bytes": g.data_bytes,
            "block_table_bytes": g.padded_bytes(),
            "effective_bits_per_weight": g.data_bytes * 8 / g.parameters,
            "by_type": by_type,
            "weight_matrices": len(weights),
            "matrices_with_row_length_multiple_of_256": len(divisible),
            "k_quant_matrices": sum(1 for t in weights if t.type_name in k_quant),
            "k_quant_exactly_on_multiples_of_256": all(
                (t.type_name in k_quant) == (t.dims[0] % 256 == 0) for t in weights
            ),
        },
        "kv": kv_rows(qwen, launches, summary),
        "published": published_rows(models, pub),
        "scenarios": scenario_rows(models),
        "unusable_launches": [
            {"config": kvcheck.config_key(x["context"], x["cache_type"]), "repeat": x["repeat"]}
            for x in launches
            if kvcheck.launch_problems(x)
        ],
    }


def kv_rows(qwen: Model, launches: Sequence[Mapping], summary: Mapping) -> list[dict]:
    rows = []
    for cache_type in kvcheck.CACHE_TYPES:
        for context in kvcheck.CONTEXTS:
            key = kvcheck.config_key(context, cache_type)
            if key not in summary:
                continue
            good = [
                x
                for x in launches
                if (x["context"], x["cache_type"]) == (context, cache_type)
                and not kvcheck.launch_problems(x)
            ]
            cells = int(_med(summary[key]["cells"]))
            reported = _med(summary[key]["kv_mib"])
            at_cells = kvmem.kv_cache_bytes(qwen, cells, cache_type=cache_type) / MIB
            at_asked = kvmem.kv_cache_bytes(qwen, context, cache_type=cache_type) / MIB
            rows.append(
                {
                    "key": key,
                    "context": context,
                    "cache_type": cache_type,
                    "launches": len(good),
                    "cells": cells,
                    "reported_mib": reported,
                    "calculator_at_cells_mib": at_cells,
                    "calculator_at_asked_mib": at_asked,
                    "difference_percent": (reported - at_cells) / at_cells * 100,
                    "k_mib": good[0]["parsed"]["kv"][0]["k_mib"],
                    "v_mib": good[0]["parsed"]["kv"][0]["v_mib"],
                    "model_mib": _med(summary[key]["model_mib"]),
                    "context_mib": _med(summary[key]["context_mib"]),
                    "compute_mib": _med([x["parsed"]["breakdown"]["compute_mib"] for x in good]),
                }
            )
    return rows


def published_rows(models: Mapping[str, Model], pub: Mapping) -> list[dict]:
    rows = []
    for name, figures in pub.items():
        counts = kvmem.param_counts(models[name])
        row = {
            "name": name,
            "total_b": counts.total / 1e9,
            "total_published": figures["total_b"],
            "total_decimals": figures["total_decimals"],
            "total_held": kvcheck.check_published(
                {name: models[name]}, {name: figures}, active=False
            )
            == [],
            "non_embedding_b": counts.non_embedding / 1e9,
            "non_embedding_published": figures.get("non_embedding_b"),
            "non_embedding_decimals": figures.get("non_embedding_decimals"),
            "active_b": counts.active / 1e9,
            "active_published": figures.get("active_b"),
            "active_decimals": figures.get("active_decimals"),
        }
        row["active_held"] = (
            None
            if "active_b" not in figures
            else kvcheck.check_published({name: models[name]}, {name: figures}, active=True) == []
        )
        rows.append(row)
    return rows


def scenario_rows(models: Mapping[str, Model]) -> list[dict]:
    rows = []
    for name, m in models.items():
        modes = ("compressed", "expanded") if m.mla else ("compressed",)
        for mode in modes:
            per_token = kvmem.kv_bytes_per_token(m, "f16", mla=mode)
            rows.append(
                {
                    "name": name + (f" ({mode} cache)" if m.mla else ""),
                    "weights_gib": kvmem.weight_bytes(kvmem.param_counts(m).total, SCENARIO_BITS)
                    / GIB,
                    "kv_per_token_bytes": per_token,
                    "kv_gib": {
                        str(c): kvmem.kv_cache_bytes(m, c, mla=mode) / GIB
                        for c in SCENARIO_CONTEXTS
                    },
                }
            )
    return rows


# ------------------------------------------------------------------- tables


def render(a: Mapping) -> str:
    return "\n\n".join(
        [
            "### Weights: the calculator against the model file",
            parameters_table(a),
            quant_table(a),
            "### KV cache: the calculator against what llama.cpp reports",
            kv_table(a),
            "### Memory breakdown llama.cpp prints at shutdown",
            breakdown_table(a),
            "### Parameters: the calculator against the published figures",
            published_table(a),
            "### What the calculator says for the five models (a calculation, not a measurement)",
            scenario_table(a),
            "### The ten predictions",
            verdict_table(a),
        ]
    )


def parameters_table(a: Mapping) -> str:
    p = a["parameters"]
    printed = p["llama_printed_millions"]
    rows = [
        ("Calculator, embeddings tied as the config says", f"{p['calculator_tied']:,}"),
        (
            "Calculator, output matrix counted separately (added after prediction 1 failed)",
            f"{p['calculator_output_matrix_counted']:,}",
        ),
        ("Elements in the GGUF file's 291 tensors", f"{p['file_elements']:,}"),
        ("llama.cpp's printed `model params`", f"{printed} M" if printed is not None else "n/a"),
        ("Vocabulary x hidden size (one embedding matrix)", f"{p['vocab_times_hidden']:,}"),
    ]
    return md(["Quantity", "Parameters"], rows)


def quant_table(a: Mapping) -> str:
    f = a["file"]
    rows = []
    for name, row in sorted(f["by_type"].items(), key=lambda kv: -kv[1]["bytes"]):
        rows.append(
            (
                name,
                row["tensors"],
                f"{row['elements']:,}",
                f"{row['bytes']:,}",
                f"{row['bytes'] * 8 / row['elements']:.2f}",
            )
        )
    rows.append(
        (
            "All",
            f["tensors"],
            f"{a['parameters']['file_elements']:,}",
            f"{f['data_bytes']:,}",
            f"{f['effective_bits_per_weight']:.2f}",
        )
    )
    return md(["Type", "Tensors", "Elements", "Bytes", "Bits per weight"], rows)


def kv_table(a: Mapping) -> str:
    rows = []
    for r in a["kv"]:
        rows.append(
            (
                r["context"],
                r["cache_type"],
                r["cells"],
                f"{r['reported_mib']:.2f}",
                f"{r['calculator_at_cells_mib']:.2f}",
                f"{r['difference_percent']:+.2f}%",
                f"{r['calculator_at_asked_mib']:.2f}",
                r["launches"],
            )
        )
    return md(
        [
            "Context asked",
            "Cache type",
            "Cells reported",
            "llama.cpp KV (MiB)",
            "Calculator at those cells (MiB)",
            "Difference",
            "Calculator at the context asked (MiB)",
            "Launches",
        ],
        rows,
    )


def breakdown_table(a: Mapping) -> str:
    rows = [
        (
            r["context"],
            r["cache_type"],
            f"{r['model_mib']:.0f}",
            f"{r['context_mib']:.0f}",
            f"{r['compute_mib']:.0f}",
            f"{r['reported_mib']:.2f}",
        )
        for r in a["kv"]
    ]
    return md(
        [
            "Context asked",
            "Cache type",
            "Model (MiB)",
            "Context (MiB)",
            "Compute (MiB)",
            "KV line (MiB)",
        ],
        rows,
    )


def _fmt(value, decimals: int | None) -> str:
    return "n/a" if value is None else f"{value:.{decimals if decimals is not None else 2}f}"


def published_table(a: Mapping) -> str:
    rows = []
    for r in a["published"]:
        total = _fmt(r["total_b"], 4)
        active = "n/a" if r["active_published"] is None else _fmt(r["active_b"], 4)
        rows.append(
            (
                r["name"],
                total,
                f"{r['total_published']}",
                "yes" if r["total_held"] else "no",
                active,
                "n/a" if r["active_published"] is None else f"{r['active_published']}",
                "n/a" if r["active_held"] is None else ("yes" if r["active_held"] else "no"),
                _fmt(r["non_embedding_b"], 4),
                "n/a"
                if r["non_embedding_published"] is None
                else f"{r['non_embedding_published']}",
            )
        )
    return md(
        [
            "Model",
            "Total, calculator (B)",
            "Total, published (B)",
            "Within rounding",
            "Active, calculator (B)",
            "Active, published (B)",
            "Within rounding ",
            "Non-embedding, calculator (B)",
            "Non-embedding, published (B)",
        ],
        rows,
    )


def scenario_table(a: Mapping) -> str:
    headers = ["Model", "Weights at 16 bits (GiB)", "KV per token, f16 (KiB)"] + [
        f"KV at {c:,} tokens (GiB)" for c in SCENARIO_CONTEXTS
    ]
    rows = [
        (
            r["name"],
            f"{r['weights_gib']:.2f}",
            f"{r['kv_per_token_bytes'] / 1024:.1f}",
            *[f"{r['kv_gib'][str(c)]:.2f}" for c in SCENARIO_CONTEXTS],
        )
        for r in a["scenarios"]
    ]
    return md(headers, rows)


def verdict_table(a: Mapping) -> str:
    rows = []
    for v in a["verdicts"]:
        detail = "held" if v["held"] else "; ".join(v["failures"])
        rows.append((v["number"], v["prediction"], "yes" if v["held"] else "no", detail))
    return md(["#", "Prediction", "Held", "Detail"], rows)
