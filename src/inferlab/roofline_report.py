"""Day 3's analysis and tables, rendered from result files, and the roofline chart as plain SVG.

Choices made before the first measurement:
- the measured bandwidth roof is the highest median among the stable streaming kernels, the measured
  single- and half-precision compute roofs are the highest medians among stable square products of
  that precision, and a number from a configuration that was not repeatable or not validated is
  never used, only listed as not citable;
- every measured point is placed against the roof of its own precision;
- a prediction whose inputs were not citable is reported as not evaluated, never as passed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from xml.sax.saxutils import escape

from . import roofline as rl

NOT_CITABLE = "not citable (runs were not repeatable or the result was wrong)"
STREAM_KERNELS = ("scale", "add", "triad", "sum")
DATACENTRE_COMPUTE = "fp16_dense"
SKINNY_SHAPE = 8192  # the decode sweep multiplies by a 8192 x 8192 half-precision matrix


def citable(summary: Mapping, key: str) -> bool:
    entry = summary.get(key)
    return bool(entry) and entry["stable"] and entry["validated"]


def measured_roofs(bandwidth: Mapping, matmul: Mapping) -> dict:
    """The roofs this chip's GPU actually reached, or None where nothing citable was measured."""

    def best(summary, keys, field):
        values = [summary[k][field]["median"] for k in keys if citable(summary, k)]
        return max(values) if values else None

    bw = best(bandwidth["summary"], STREAM_KERNELS, "bytes_per_s")
    fp32 = [k for k in matmul["summary"] if k.startswith("fp32-")]
    fp16 = [k for k in matmul["summary"] if k.startswith("fp16-")]
    return {
        "bandwidth": bw,
        "fp32": best(matmul["summary"], fp32, "flops_per_s"),
        "fp16": best(matmul["summary"], fp16, "flops_per_s"),
    }


def cpu_roofs(microbench: Mapping, compute: Mapping) -> dict:
    """The CPU's roofs from Day 2: best stable triad median, best stable Accelerate median."""
    configs = microbench["groups"]["bandwidth"]["configs"]
    triads = [
        c["kernels"]["triad"]["median"]
        for key, c in configs.items()
        if key.startswith("interactive-") and c["kernels"]["triad"]["stable"]
    ]
    summary = compute["groups"]["compute"]["summary"]
    gflops = [
        v["median"]
        for key, v in summary.items()
        if key.startswith("default/accelerate/") and v["stable"]
    ]
    if not (triads and gflops):
        raise ValueError("Day 2 has no stable triad or Accelerate result to take a CPU roof from")
    return {"peak_flops": max(gflops) * 1e9, "bandwidth": max(triads) * 1e9}


def analyze(bandwidth: Mapping, matmul: Mapping, skinny: Mapping) -> dict:
    """Roofs, the predicted decode knee, every placed point, and the prediction verdicts."""
    roofs = measured_roofs(bandwidth, matmul)
    bw_summary, mm, sk = bandwidth["summary"], matmul["summary"], skinny["summary"]
    verdicts: dict[str, list[str] | None] = {"bandwidth": None, "matmul": None, "skinny": None}
    if citable(bw_summary, "triad"):
        verdicts["bandwidth"] = rl.check_bandwidth(
            bw_summary["triad"]["bytes_per_s"]["median"] / 1e9
        )
    needed = ("fp32-4096", "fp16-4096", "fp32-512")
    if all(citable(mm, k) for k in needed):
        verdicts["matmul"] = rl.check_matmul(
            mm["fp32-4096"]["flops_per_s"]["median"] / 1e12,
            mm["fp16-4096"]["flops_per_s"]["median"] / 1e12,
            mm["fp32-512"]["flops_per_s"]["median"] / 1e12,
        )

    knee = None
    points: list[dict] = []
    gpu = {p: roofs[p] for p in ("bandwidth", "fp32", "fp16")}
    if all(gpu.values()):
        roof16 = rl.Roof("GPU fp16", gpu["fp16"], gpu["bandwidth"])
        roof32 = rl.Roof("GPU fp32", gpu["fp32"], gpu["bandwidth"])
        first = next(iter(sk.values()))
        k, n = first["k"], first["n"]
        knee = rl.knee_batch(k, n, 2, 2, 2, roof16.ridge)
        for series, summary, keys in (
            ("stream", bw_summary, STREAM_KERNELS),
            ("matmul fp32", mm, [x for x in mm if x.startswith("fp32-")]),
            ("matmul fp16", mm, [x for x in mm if x.startswith("fp16-")]),
            ("matmul bf16", mm, [x for x in mm if x.startswith("bf16-")]),
            ("decode batch", sk, list(sk)),
        ):
            roof = roof32 if series == "matmul fp32" else roof16
            for key in keys:
                if not citable(summary, key):
                    continue
                entry = summary[key]
                placed = rl.place(roof, entry["intensity"], entry["flops_per_s"]["median"])
                points.append(
                    {
                        "series": series,
                        "key": key,
                        "bytes_per_s": entry["bytes_per_s"]["median"],
                        **placed,
                    }
                )
        decode = [
            {
                "batch": int(key.split("-")[1]),
                "intensity": sk[key]["intensity"],
                "flops_per_s": sk[key]["flops_per_s"]["median"],
                "bytes_per_s": sk[key]["bytes_per_s"]["median"],
            }
            for key in sk
            if citable(sk, key)
        ]
        verdicts["skinny"] = rl.check_skinny(decode, roof16, knee)
        if len(decode) < len(sk):
            verdicts["skinny"].append(f"{len(sk) - len(decode)} batch size(s) were not citable")
    return {"roofs": roofs, "knee_batch": knee, "points": points, "verdicts": verdicts}


def _cell(entry: Mapping, field: str, scale: float, digits: int = 1) -> str:
    stat = entry[field]
    low, mid, high = (stat[k] / scale for k in ("min", "median", "max"))
    return f"{mid:.{digits}f} ({low:.{digits}f} to {high:.{digits}f})"


def spec_table(specs: Mapping) -> str:
    """The specification figures as printed, with derived ones marked, and each device's ridge."""
    lines = [
        "| Device | Memory bandwidth | FP16/BF16 tensor, dense | FP32 | Ridge, FLOP/byte | "
        "Decode batch to reach the ridge (fp16 weights) |",
        "|---|---|---|---|---|---|",
    ]

    def show(fig: Mapping | None) -> str:
        if not fig:
            return "not published by the vendor"
        mark = " (derived)" if fig["how"] == "derived" else ""
        bound = "up to " if str(fig.get("as_printed", "")).lower().startswith("up to") else ""
        return f"{bound}{fig['value']:g} {fig['unit']}{mark}"

    for device in specs["devices"]:
        compute = device.get("compute") or {}
        roof = rl.spec_roof(device, DATACENTRE_COMPUTE)
        ridge = f"{roof.ridge:.0f}" if roof else "not published"
        knee = rl.knee_batch(SKINNY_SHAPE, SKINNY_SHAPE, 2, 2, 2, roof.ridge) if roof else None
        batch = f"{knee:.0f}" if knee else "not published" if roof is None else "never"
        lines.append(
            f"| {device['name']} | {show(device['memory_bandwidth'])} | "
            f"{show(compute.get(DATACENTRE_COMPUTE))} | {show(compute.get('fp32'))} | "
            f"{ridge} | {batch} |"
        )
    return "\n".join(lines)


def bandwidth_table(result: Mapping) -> str:
    lines = [
        "| Kernel | Bytes moved per element | GB/s, median (min to max) | Variation |",
        "|---|---|---|---|",
    ]
    per_element = {"scale": 8, "add": 12, "triad": 12, "sum": 4}
    for key, entry in result["summary"].items():
        if not citable(result["summary"], key):
            lines.append(f"| {key} | {per_element.get(key, '')} | {NOT_CITABLE} | |")
            continue
        cv = entry["bytes_per_s"]["cv"]
        lines.append(
            f"| {key} | {per_element.get(key, '')} | "
            f"{_cell(entry, 'bytes_per_s', 1e9)} | {cv:.1%} |"
        )
    return "\n".join(lines)


def matmul_table(result: Mapping) -> str:
    sizes = sorted({int(k.split("-")[1]) for k in result["summary"]})
    dtypes = ["fp32", "fp16", "bf16"]
    lines = [
        "| TFLOPS, median (min to max) | " + " | ".join(f"n={n}" for n in sizes) + " |",
        "|---|" + "---|" * len(sizes),
    ]
    for dtype in dtypes:
        cells = []
        for n in sizes:
            key = f"{dtype}-{n}"
            if key not in result["summary"]:
                cells.append("")
            elif not citable(result["summary"], key):
                cells.append("not citable")
            else:
                cells.append(_cell(result["summary"][key], "flops_per_s", 1e12, 2))
        lines.append(f"| {dtype} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def placement_table(analysis: Mapping) -> str:
    lines = [
        "| Series | Point | Intensity, FLOP/byte | Achieved GFLOPS | Roofline GFLOPS | "
        "Fraction of roofline | Bound |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in analysis["points"]:
        lines.append(
            f"| {p['series']} | {p['key']} | {p['intensity']:.3g} "
            f"| {p['achieved_flops'] / 1e9:.0f} "
            f"| {p['attainable_flops'] / 1e9:.0f} | {p['fraction_of_roof']:.0%} | {p['bound']} |"
        )
    return "\n".join(lines)


def roofs_table(analysis: Mapping, cpu: Mapping) -> str:
    roofs = analysis["roofs"]

    def row(name, peak, bw, source):
        if not (peak and bw):
            return f"| {name} | {NOT_CITABLE} | | | {source} |"
        return f"| {name} | {peak / 1e12:.2f} | {bw / 1e9:.1f} | {peak / bw:.1f} | {source} |"

    lines = [
        "| Roof | Peak TFLOPS | Bandwidth GB/s | Ridge, FLOP/byte | Where it comes from |",
        "|---|---|---|---|---|",
        row("GPU, half precision", roofs["fp16"], roofs["bandwidth"], "measured today through MLX"),
        row(
            "GPU, single precision", roofs["fp32"], roofs["bandwidth"], "measured today through MLX"
        ),
        row("CPU, single precision", cpu["peak_flops"], cpu["bandwidth"], "measured on Day 2"),
    ]
    return "\n".join(lines)


def decode_table(analysis: Mapping, skinny: Mapping) -> str:
    lines = [
        "| Batch | Intensity, FLOP/byte | TFLOPS | Weight traffic, GB/s | "
        "Fraction of roofline | Bound |",
        "|---|---|---|---|---|---|",
    ]
    placed = {p["key"]: p for p in analysis["points"] if p["series"] == "decode batch"}
    for key in skinny["summary"]:
        batch = key.split("-")[1]
        if key not in placed:
            lines.append(f"| {batch} | {NOT_CITABLE} | | | | |")
            continue
        p = placed[key]
        lines.append(
            f"| {batch} | {p['intensity']:.3g} | {p['achieved_flops'] / 1e12:.2f} "
            f"| {p['bytes_per_s'] / 1e9:.1f} | {p['fraction_of_roof']:.0%} | {p['bound']} |"
        )
    return "\n".join(lines)


def predictions_table(analysis: Mapping) -> str:
    lines = ["| Prediction | Result |", "|---|---|"]
    names = {
        "bandwidth": "Triad bandwidth between 60 and 100 GB/s",
        "matmul": "fp32 n=4096 between 1.0 and 4.0 TFLOPS; fp16 at least 0.9 of it; "
        "n=4096 at least 1.2 times n=512",
        "skinny": "Decode sweep: never above its roofline; memory-bound and compute-bound "
        "regions reach their floors",
    }
    for key, text in names.items():
        verdict = analysis["verdicts"][key]
        if verdict is None:
            result = "not evaluated: its inputs were not citable"
        elif verdict:
            result = "did not hold: " + "; ".join(verdict)
        else:
            result = "held"
        lines.append(f"| {text} | {result} |")
    return "\n".join(lines)


# ---- chart -------------------------------------------------------------------------------------

X_RANGE = (0.05, 4096.0)  # FLOP per byte
Y_RANGE = (1e9, 1e16)  # FLOP per second
WIDTH, HEIGHT = 980, 580
LEFT, RIGHT, TOP, BOTTOM = 100, 280, 24, 58
PALETTE = ["#d1495b", "#edae49", "#66a182", "#00798c", "#8e6bbf", "#30638e", "#b0813f", "#6c757d"]
SERIES_COLOUR = {
    "stream": "#2a9d8f",
    "matmul fp32": "#e76f51",
    "matmul fp16": "#264653",
    "matmul bf16": "#8d6a9f",
    "decode batch": "#d62828",
}


def log_position(value: float, low: float, high: float, start: float, end: float) -> float:
    """Where `value` falls on a log axis running from `start` (at `low`) to `end` (at `high`)."""
    fraction = (math.log10(value) - math.log10(low)) / (math.log10(high) - math.log10(low))
    return start + fraction * (end - start)


def _px(intensity: float) -> float:
    return log_position(intensity, *X_RANGE, LEFT, WIDTH - RIGHT)


def _py(flops: float) -> float:
    return log_position(flops, *Y_RANGE, HEIGHT - BOTTOM, TOP)


def roof_polyline(peak: float, bandwidth: float) -> list[tuple[float, float]]:
    """Corner points of one roof, clipped to the chart: a rising edge, then a flat top."""
    start_x = max(X_RANGE[0], Y_RANGE[0] / bandwidth)
    ridge = peak / bandwidth
    return [
        (_px(start_x), _py(bandwidth * start_x)),
        (_px(ridge), _py(peak)),
        (_px(X_RANGE[1]), _py(peak)),
    ]


def roofline_svg(roofs: Sequence[Mapping], points: Sequence[Mapping]) -> str:
    """Log-log roofline chart. Each roof is {name, peak_flops, bandwidth, dashed?, bold?}."""
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" '
        f'aria-label="Roofline chart: attainable FLOP per second against FLOP per byte">',
        "<style>:root{--bg:#ffffff;--fg:#24292f;--grid:#d8dee4}"
        "@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--grid:#30363d}}"
        "text{fill:var(--fg);font:12px system-ui,sans-serif}"
        ".grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--fg);stroke-width:1.2}"
        ".roof{fill:none;stroke-width:2}</style>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="var(--bg)"/>',
    ]
    for exp in range(-1, 4):
        value = 10.0**exp
        if X_RANGE[0] <= value <= X_RANGE[1]:
            x = _px(value)
            parts.append(
                f'<line class="grid" x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{HEIGHT - BOTTOM}"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{HEIGHT - BOTTOM + 18}" '
                f'text-anchor="middle">{value:g}</text>'
            )
    x = _px(X_RANGE[1])
    parts.append(f'<text x="{x:.1f}" y="{HEIGHT - BOTTOM + 18}" text-anchor="end">4096</text>')
    for exp in range(9, 17):
        y = _py(10.0**exp)
        parts.append(
            f'<line class="grid" x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH - RIGHT}" y2="{y:.1f}"/>'
        )
        label = {9: "1 GFLOPS", 10: "10 GFLOPS", 11: "100 GFLOPS", 12: "1 TFLOPS", 13: "10 TFLOPS",
                 14: "100 TFLOPS", 15: "1 PFLOPS", 16: "10 PFLOPS"}[exp]  # fmt: skip
        parts.append(f'<text x="{LEFT - 8}" y="{y + 4:.1f}" text-anchor="end">{label}</text>')
    parts.append(
        f'<line class="axis" x1="{LEFT}" y1="{HEIGHT - BOTTOM}" '
        f'x2="{WIDTH - RIGHT}" y2="{HEIGHT - BOTTOM}"/>'
    )
    parts.append(f'<line class="axis" x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{HEIGHT - BOTTOM}"/>')
    parts.append(
        f'<text x="{(LEFT + WIDTH - RIGHT) / 2:.0f}" y="{HEIGHT - 14}" text-anchor="middle">'
        "arithmetic intensity, FLOP per byte moved</text>"
    )
    parts.append(
        f'<text transform="translate(16 {(TOP + HEIGHT - BOTTOM) / 2:.0f}) rotate(-90)" '
        'text-anchor="middle">attainable performance</text>'
    )

    legend_y = TOP + 6
    for index, roof in enumerate(roofs):
        colour = roof.get("colour") or PALETTE[index % len(PALETTE)]
        corners = " ".join(
            f"{x:.1f},{y:.1f}" for x, y in roof_polyline(roof["peak_flops"], roof["bandwidth"])
        )
        dash = ' stroke-dasharray="7 5"' if roof.get("dashed") else ""
        width = ' style="stroke-width:3.5"' if roof.get("bold") else ""
        parts.append(f'<polyline class="roof" stroke="{colour}"{dash}{width} points="{corners}"/>')
        lx = WIDTH - RIGHT + 14
        parts.append(
            f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 22}" y2="{legend_y}" stroke="{colour}" '
            f'stroke-width="2.5"{dash}/>'
        )
        parts.append(f'<text x="{lx + 28}" y="{legend_y + 4}">{escape(roof["name"])}</text>')
        legend_y += 20
    legend_y += 8
    seen: set[str] = set()
    for p in points:
        colour = SERIES_COLOUR.get(p["series"], "#555555")
        parts.append(
            f'<circle cx="{_px(p["intensity"]):.1f}" cy="{_py(p["achieved_flops"]):.1f}" r="4.5" '
            f'fill="{colour}" fill-opacity="0.85">'
            f"<title>{escape(p['series'])} {escape(p['key'])}</title></circle>"
        )
        if p["series"] not in seen:
            seen.add(p["series"])
            lx = WIDTH - RIGHT + 14
            parts.append(f'<circle cx="{lx + 11}" cy="{legend_y}" r="4.5" fill="{colour}"/>')
            parts.append(
                f'<text x="{lx + 28}" y="{legend_y + 4}">measured: {escape(p["series"])}</text>'
            )
            legend_y += 20
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
