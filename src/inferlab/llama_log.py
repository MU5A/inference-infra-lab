"""Read the memory figures out of a llama-server log, as llama.cpp (build cbb1c1c) prints them.

The server is started with `--log-jsonl`, so stdout is one JSON object per line. Messages come as
`{"type": "log", "msg": "..."}`, and the memory breakdown printed at shutdown also arrives as a
structured `{"type": "fit_memory_breakdown", "data": {"unit": "MiB", "rows": [...]}}` event. The
event is printed once while the context is sized and again at clean shutdown, so the LAST one is
the one that describes the running server.

Nothing here raises on a line it does not understand. A figure that is missing is None or an empty
list, and the caller decides whether that is a failed run.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

_NUMBER = r"([0-9]+(?:\.[0-9]+)?)"
KV_LINE = re.compile(
    r"llama_kv_cache: size =\s*" + _NUMBER + r" MiB \(\s*(\d+) cells,\s*(\d+) layers,"
    r"\s*(\d+)/(\d+) seqs\), K \((\w+)\):\s*"
    + _NUMBER
    + r" MiB, V \((\w+)\):\s*"
    + _NUMBER
    + r" MiB"
)
PARAMS_LINE = re.compile(r"print_info: model params\s*=\s*" + _NUMBER + r" ([KMBT])\b")
FILE_LINE = re.compile(r"print_info: file size\s*=\s*" + _NUMBER + r" MiB \(" + _NUMBER + r" BPW\)")
MODEL_BUFFER = re.compile(r"load_tensors:\s+(\S+) model buffer size =\s*" + _NUMBER + r" MiB")
COMPUTE_BUFFER = re.compile(r"sched_reserve:\s+(\S+) compute buffer size =\s*" + _NUMBER + r" MiB")
OUTPUT_BUFFER = re.compile(r"llama_context:\s+(\S+)\s+output buffer size =\s*" + _NUMBER + r" MiB")
CONTEXT_LINE = re.compile(r"llama_context: n_ctx\s*=\s*(\d+)")
FLASH_ON = re.compile(r"Flash Attention enabled")  # printed when `-fa auto` resolves to on
FLASH_LINE = re.compile(r"llama_context: flash_attn\s*=\s*(\w+)")  # `-fa on` prints "enabled"
_TO_MILLIONS = {"K": 1e-3, "M": 1.0, "B": 1e3, "T": 1e6}


def _messages(lines: Iterable[str]) -> Iterable[dict]:
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


def parse(lines: Iterable[str]) -> dict:
    kv, model_buffers, compute_buffers, output_buffers = [], [], [], []
    params = file_size = bpw = n_ctx = None
    flash = False
    breakdowns = []
    for event in _messages(lines):
        if event.get("type") == "fit_memory_breakdown":
            data = event.get("data")
            if isinstance(data, dict) and isinstance(data.get("rows"), list):
                breakdowns.append(data["rows"])
            continue
        msg = event.get("msg")
        if not isinstance(msg, str):
            continue
        if m := KV_LINE.search(msg):
            kv.append(
                {
                    "size_mib": float(m[1]),
                    "cells": int(m[2]),
                    "layers": int(m[3]),
                    "seqs_used": int(m[4]),
                    "seqs_total": int(m[5]),
                    "k_type": m[6],
                    "k_mib": float(m[7]),
                    "v_type": m[8],
                    "v_mib": float(m[9]),
                }
            )
        elif m := PARAMS_LINE.search(msg):
            params = float(m[1]) * _TO_MILLIONS[m[2]]
        elif m := FILE_LINE.search(msg):
            file_size, bpw = float(m[1]), float(m[2])
        elif m := MODEL_BUFFER.search(msg):
            model_buffers.append({"buffer": m[1], "mib": float(m[2])})
        elif m := COMPUTE_BUFFER.search(msg):
            compute_buffers.append({"buffer": m[1], "mib": float(m[2])})
        elif m := OUTPUT_BUFFER.search(msg):
            output_buffers.append({"buffer": m[1], "mib": float(m[2])})
        elif m := CONTEXT_LINE.search(msg):
            n_ctx = int(m[1])
        elif m := FLASH_LINE.search(msg):
            flash = flash or m[1] in ("enabled", "on")
        elif FLASH_ON.search(msg):
            flash = True
    return {
        "kv": kv,
        "model_params_millions": params,
        "file_size_mib": file_size,
        "bits_per_weight": bpw,
        "n_ctx": n_ctx,
        "flash_attention_enabled": flash,
        "model_buffers": model_buffers,
        "compute_buffers": compute_buffers,
        "output_buffers": output_buffers,
        "breakdown_events": len(breakdowns),
        "breakdown": summarize_breakdown(breakdowns[-1]) if breakdowns else None,
    }


def summarize_breakdown(rows: list) -> dict:
    """Column totals across the rows of one breakdown, in MiB, plus the rows themselves."""
    clean = [r for r in rows if isinstance(r, dict)]

    def total(column: str) -> int:
        return sum(r[column] for r in clean if isinstance(r.get(column), int))

    return {
        "model_mib": total("model"),
        "context_mib": total("context"),
        "compute_mib": total("compute"),
        "rows": clean,
    }
