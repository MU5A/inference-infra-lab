"""Day 4's tables, rendered from the result files so the write-up cannot drift from them.

Rules, fixed before the first measurement:
- a configuration is citable only if its runs repeated within the limit, its output validated, and
  every run says it ran at the threadgroup size asked for; any other configuration is shown as not
  citable, with no number;
- a prediction is judged only when every configuration it reads is citable, otherwise it is
  reported as not evaluated, never as held.
"""

from __future__ import annotations

from collections.abc import Mapping

from . import metal_sweep as ms

NOT_CITABLE = "not citable"


def citable(entry: Mapping | None) -> bool:
    return bool(entry) and entry["stable"] and entry["validated"] and entry["dispatched_as_asked"]


def _cell(entry: Mapping | None, field: str, scale: float, digits: int) -> str:
    if not citable(entry):
        return NOT_CITABLE
    stat = entry[field]
    low, mid, high = (stat[k] / scale for k in ("min", "median", "max"))
    return f"{mid:.{digits}f} ({low:.{digits}f} to {high:.{digits}f})"


def _cv(entry: Mapping | None, field: str) -> str:
    if not entry or entry[field]["cv"] is None:
        return "n/a"
    return f"{entry[field]['cv']:.1%}"


def stream_table(stream: Mapping, mlx_triad: float) -> str:
    s = stream["summary"]
    lines = [
        "| Threadgroup size | scalar, GB/s median (min to max) | scalar, % of MLX triad "
        "| vec4, GB/s median (min to max) | vec4, % of MLX triad |",
        "|---|---|---|---|---|",
    ]
    for tg in ms.THREADGROUP_SIZES:
        cells = []
        for variant in ms.STREAM_VARIANTS:
            entry = s.get(ms.stream_key(variant, tg))
            cells.append(_cell(entry, "bytes_per_s", 1e9, 1))
            share = entry["bytes_per_s"]["median"] / mlx_triad if citable(entry) else None
            cells.append(f"{share:.0%}" if share is not None else NOT_CITABLE)
        lines.append(f"| {tg} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def occupancy_table(occupancy: Mapping) -> str:
    s = occupancy["summary"]
    base = s.get(ms.occupancy_key(ms.SCRATCH_BYTES[0]))
    lines = [
        "| Threadgroup memory per group of "
        f"{ms.OCCUPANCY_THREADGROUP} | GB/s median (min to max) | Variation "
        f"| Relative to {ms.SCRATCH_BYTES[0]} bytes |",
        "|---|---|---|---|",
    ]
    for size in ms.SCRATCH_BYTES:
        entry = s.get(ms.occupancy_key(size))
        if citable(entry) and citable(base):
            ratio = entry["bytes_per_s"]["median"] / base["bytes_per_s"]["median"]
            relative = f"{ratio:.2f}"
        else:
            relative = NOT_CITABLE
        label = f"{size // 1024} KiB"
        lines.append(
            f"| {label} | {_cell(entry, 'bytes_per_s', 1e9, 1)} | {_cv(entry, 'bytes_per_s')} "
            f"| {relative} |"
        )
    return "\n".join(lines)


def fma_table(fma: Mapping) -> str:
    s = fma["summary"]
    lines = [
        "| Threadgroup size | fp32 TFLOPS, median (min to max) | fp32 variation "
        "| fp16 TFLOPS, median (min to max) | fp16 variation |",
        "|---|---|---|---|---|",
    ]
    for tg in ms.THREADGROUP_SIZES:
        cells = []
        for dtype in ms.FMA_DTYPES:
            entry = s.get(ms.fma_key(dtype, tg))
            cells += [_cell(entry, "flops_per_s", 1e12, 2), _cv(entry, "flops_per_s")]
        lines.append(f"| {tg} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def dispatch_table(results: Mapping[str, Mapping]) -> str:
    """What the kernels themselves reported about how they ran."""
    lines = [
        "| Phase | Configurations | Ran at the threadgroup size asked, in every run "
        "| SIMD-group width reported |",
        "|---|---|---|---|",
    ]
    for phase, result in results.items():
        s = result["summary"]
        asked = sum(e["dispatched_as_asked"] for e in s.values())
        widths = sorted({w for e in s.values() for w in e["simd_widths"]})
        shown = ", ".join(str(w) for w in widths) or "none reported"
        lines.append(f"| {phase} | {len(s)} | {asked} of {len(s)} | {shown} |")
    return "\n".join(lines)


def judge(stream: Mapping, occupancy: Mapping, fma: Mapping, mlx_triad, mlx_fp32) -> list[dict]:
    """The prediction verdicts, with any phase that has a non-citable configuration not judged."""
    verdicts = ms.verdicts(
        stream["summary"], occupancy["summary"], fma["summary"], mlx_triad, mlx_fp32
    )
    phases = {"stream": stream, "occupancy": occupancy, "fma": fma}
    for v in verdicts:
        summary = phases[v["phase"]]["summary"]
        if not all(citable(e) for e in summary.values()):
            v["held"] = None
            v["failures"] = []
    return verdicts


def predictions_table(verdicts: list[dict]) -> str:
    lines = ["| Prediction | Result |", "|---|---|"]
    for v in verdicts:
        if v["held"] is None:
            result = "not evaluated: a configuration it reads was not citable"
        elif v["held"]:
            result = "held"
        else:
            result = "did not hold: " + "; ".join(v["failures"])
        lines.append(f"| {v['prediction']} | {result} |")
    return "\n".join(lines)
