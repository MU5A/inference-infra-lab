"""Day 3's analysis and tables, built from small result files worked out by hand."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from inferlab import roofline as rl
from inferlab import roofline_report as rr


def stat(median, cv=0.01):
    return {"median": median, "min": median * 0.99, "max": median * 1.01, "cv": cv, "runs": 3}


def cfg(flops, moved, rate_flops, stable=True, validated=True, **extra):
    """A summarised configuration with a given median FLOP/s; bytes/s follows from the ratio."""
    return {
        "flops_per_call": flops,
        "bytes_per_call": moved,
        "intensity": flops / moved,
        "flops_per_s": stat(rate_flops),
        "bytes_per_s": stat(rate_flops * moved / flops),
        "stable": stable,
        "validated": validated,
        **extra,
    }


def bandwidth_result(triad_gbs=80.0, **overrides):
    n = 1000
    out = {
        "scale": cfg(n, 8 * n, triad_gbs * 1e9 * 0.9 / 8 * 1),  # 0.9 of the triad rate in bytes/s
        "add": cfg(n, 12 * n, triad_gbs * 1e9 * 0.95 / 12),
        "triad": cfg(2 * n, 12 * n, triad_gbs * 1e9 * 2 / 12),
        "sum": cfg(n, 4 * n, triad_gbs * 1e9 * 0.9 / 4),
    }
    out.update(overrides)
    return {"summary": out}


def matmul_result(fp32=2.5e12, fp16=3.0e12, small=1.5e12, **overrides):
    def sq(n, rate):
        return cfg(2 * n**3, 3 * n * n * 2, rate, n=n)

    s = {
        "fp32-512": sq(512, small),
        "fp32-4096": sq(4096, fp32),
        "fp16-512": sq(512, small * 1.1),
        "fp16-4096": sq(4096, fp16),
        "bf16-4096": sq(4096, fp16 * 0.99),
    }
    s.update(overrides)
    return {"summary": s}


def skinny_result(rows):
    """rows: batch -> (TFLOPS, GB/s). Shape 8192 x 8192 half precision."""
    s = {}
    for batch, (tflops, gbs) in rows.items():
        flops = rl.gemm_flops(batch, 8192, 8192)
        moved = rl.gemm_bytes(batch, 8192, 8192, 2)
        s[f"batch-{batch}"] = {
            **cfg(flops, moved, tflops * 1e12, k=8192, n=8192, batch=batch),
            "bytes_per_s": stat(gbs * 1e9),
        }
    return {"summary": s}


class TestRoofs:
    def test_the_bandwidth_roof_is_the_best_stable_kernel(self):
        roofs = rr.measured_roofs(bandwidth_result(80.0), matmul_result())
        # bytes/s of each kernel: scale 72, add 76, triad 80 (and 'sum' 72): the roof is 80 GB/s.
        assert roofs["bandwidth"] == pytest.approx(80e9, rel=1e-9)

    def test_an_unstable_kernel_does_not_set_the_roof(self):
        fast_but_unstable = cfg(1000, 12000, 1e9, stable=False)
        fast_but_unstable["bytes_per_s"] = stat(500e9)
        roofs = rr.measured_roofs(bandwidth_result(80.0, add=fast_but_unstable), matmul_result())
        assert roofs["bandwidth"] == pytest.approx(80e9, rel=1e-9)

    def test_a_wrong_result_does_not_set_the_roof(self):
        wrong = cfg(1000, 12000, 1e9, validated=False)
        wrong["bytes_per_s"] = stat(500e9)
        roofs = rr.measured_roofs(bandwidth_result(80.0, add=wrong), matmul_result())
        assert roofs["bandwidth"] == pytest.approx(80e9, rel=1e-9)

    def test_compute_roofs_are_the_best_stable_square_product_of_that_precision(self):
        roofs = rr.measured_roofs(bandwidth_result(), matmul_result(fp32=2.5e12, fp16=3.0e12))
        assert roofs["fp32"] == pytest.approx(2.5e12) and roofs["fp16"] == pytest.approx(3.0e12)

    def test_nothing_citable_gives_no_roof(self):
        bad = {k: cfg(1, 1, 1e9, stable=False) for k in ("scale", "add", "triad", "sum")}
        assert rr.measured_roofs({"summary": bad}, matmul_result())["bandwidth"] is None


def sweep():
    # Roofs: 80 GB/s and 3 TFLOPS half precision, so the ridge is 37.5 and the knee about 37.7.
    return skinny_result(
        {
            1: (0.08, 80.0),
            4: (0.31, 78.0),
            8: (0.60, 75.0),
            64: (2.0, 30.0),
            256: (2.9, 12.0),
            1024: (3.0, 4.0),
        }
    )


class TestCpuRoofs:
    def micro(self, triads):
        configs = {
            f"interactive-{i}": {"kernels": {"triad": {"median": m, "stable": ok}}}
            for i, (m, ok) in enumerate(triads, start=1)
        }
        configs["background-1"] = {"kernels": {"triad": {"median": 999.0, "stable": True}}}
        return {"groups": {"bandwidth": {"configs": configs}}}

    def compute(self, rows):
        return {"groups": {"compute": {"summary": rows}}}

    def test_the_cpu_roofs_are_the_best_stable_triad_and_accelerate_medians(self):
        micro = self.micro([(82.2, True), (85.1, True), (81.3, True)])
        comp = self.compute(
            {
                "default/accelerate/128": {"median": 1303.4, "stable": True},
                "default/accelerate/512": {"median": 1653.6, "stable": True},
                "default/ikj/512": {"median": 29.9, "stable": True},
            }
        )
        roofs = rr.cpu_roofs(micro, comp)  # the background-QoS 999 and the loop 29.9 are ignored
        assert roofs["bandwidth"] == pytest.approx(85.1e9)
        assert roofs["peak_flops"] == pytest.approx(1653.6e9)

    def test_an_unstable_entry_is_not_used(self):
        micro = self.micro([(82.2, True), (99.0, False)])
        comp = self.compute({"default/accelerate/512": {"median": 1653.6, "stable": True}})
        assert rr.cpu_roofs(micro, comp)["bandwidth"] == pytest.approx(82.2e9)

    def test_no_stable_result_is_an_error_not_a_guess(self):
        micro = self.micro([(82.2, False)])
        comp = self.compute({"default/accelerate/512": {"median": 1653.6, "stable": True}})
        with pytest.raises(ValueError):
            rr.cpu_roofs(micro, comp)


class TestAnalyze:
    def test_the_knee_follows_from_the_measured_roofs(self):
        a = rr.analyze(bandwidth_result(80.0), matmul_result(), sweep())
        ridge = 3.0e12 / 80e9
        assert a["knee_batch"] == pytest.approx(rl.knee_batch(8192, 8192, 2, 2, 2, ridge))
        assert 37 < a["knee_batch"] < 38.5

    def test_good_data_passes_every_prediction(self):
        a = rr.analyze(bandwidth_result(80.0), matmul_result(), sweep())
        assert a["verdicts"] == {"bandwidth": [], "matmul": [], "skinny": []}

    def test_every_citable_point_is_placed_against_the_roof_of_its_precision(self):
        a = rr.analyze(bandwidth_result(80.0), matmul_result(), sweep())
        by_key = {(p["series"], p["key"]): p for p in a["points"]}
        fp32_point = by_key[("matmul fp32", "fp32-4096")]
        assert fp32_point["attainable_flops"] == pytest.approx(2.5e12)  # its own roof, not fp16's
        assert by_key[("matmul fp16", "fp16-4096")]["attainable_flops"] == pytest.approx(3.0e12)
        assert by_key[("decode batch", "batch-1")]["bound"] == "memory"
        assert by_key[("decode batch", "batch-1024")]["bound"] == "compute"
        assert len([p for p in a["points"] if p["series"] == "stream"]) == 4

    def test_a_slow_memory_bound_decode_point_is_reported(self):
        slow = skinny_result({1: (0.04, 40.0), 1024: (3.0, 4.0)})
        a = rr.analyze(bandwidth_result(80.0), matmul_result(), slow)
        assert any("memory-bound" in f for f in a["verdicts"]["skinny"])

    def test_a_bandwidth_outside_the_prediction_is_reported(self):
        a = rr.analyze(bandwidth_result(40.0), matmul_result(), sweep())
        assert a["verdicts"]["bandwidth"]

    def test_an_unstable_input_means_the_prediction_is_not_evaluated(self):
        unstable = cfg(2 * 1000, 12 * 1000, 1e9, stable=False)
        a = rr.analyze(bandwidth_result(80.0, triad=unstable), matmul_result(), sweep())
        assert a["verdicts"]["bandwidth"] is None

    def test_an_unstable_matmul_size_means_no_matmul_verdict(self):
        bad = cfg(2 * 512**3, 3 * 512 * 512 * 2, 1e12, stable=False, n=512)
        a = rr.analyze(bandwidth_result(80.0), matmul_result(**{"fp32-512": bad}), sweep())
        assert a["verdicts"]["matmul"] is None

    def test_an_unstable_batch_is_left_out_and_said_so(self):
        rows = sweep()
        rows["summary"]["batch-64"]["stable"] = False
        a = rr.analyze(bandwidth_result(80.0), matmul_result(), rows)
        assert not any(p["key"] == "batch-64" for p in a["points"])
        assert any("not citable" in f for f in a["verdicts"]["skinny"])


class TestTables:
    def specs(self):
        return {
            "devices": [
                {
                    "name": "Dev GPU",
                    "memory_bandwidth": {"value": 3.35, "unit": "TB/s", "how": "printed"},
                    "compute": {
                        "fp16_dense": {"value": 989.5, "unit": "TFLOPS", "how": "derived"},
                        "fp32": {"value": 67, "unit": "TFLOPS", "how": "printed"},
                    },
                },
                {
                    "name": "Dev laptop",
                    "memory_bandwidth": {"value": 100, "unit": "GB/s", "how": "printed"},
                    "compute": {},
                },
            ]
        }

    def test_spec_table_marks_derived_figures_and_gives_the_ridge(self):
        lines = rr.spec_table(self.specs()).splitlines()
        gpu = lines[2]
        assert "3.35 TB/s" in gpu and "989.5 TFLOPS (derived)" in gpu and "67 TFLOPS |" in gpu
        assert "| 295 |" in gpu  # 989.5e12 / 3.35e12 = 295.4

    def test_an_up_to_figure_is_shown_as_an_upper_bound(self):
        specs = self.specs()
        specs["devices"][0]["memory_bandwidth"]["as_printed"] = "Up to 8TB/s"
        assert "| up to 3.35 TB/s |" in rr.spec_table(specs)
        specs["devices"][0]["memory_bandwidth"]["as_printed"] = "3.35TB/s"
        assert "up to" not in rr.spec_table(specs)

    def test_spec_table_does_not_invent_figures_the_vendor_did_not_print(self):
        laptop = rr.spec_table(self.specs()).splitlines()[3]
        assert "100 GB/s" in laptop and laptop.count("not published") >= 3

    def test_bandwidth_table_hides_unstable_kernels(self):
        res = bandwidth_result(80.0, add=cfg(1000, 12000, 1e9, stable=False))
        lines = rr.bandwidth_table(res).splitlines()
        add = next(x for x in lines if x.startswith("| add"))
        assert rr.NOT_CITABLE in add
        assert "80.0 (79.2 to 80.8)" in next(x for x in lines if x.startswith("| triad"))

    def test_matmul_table_is_a_precision_by_size_grid(self):
        lines = rr.matmul_table(matmul_result(fp32=2.6e12)).splitlines()
        assert lines[0].startswith("| TFLOPS, median (min to max) | n=512 | n=4096 |")
        fp32 = next(x for x in lines if x.startswith("| fp32"))
        assert "2.60 (2.57 to 2.63)" in fp32  # 2.6 times 0.99 and 1.01, no rounding ties

    def test_matmul_table_hides_an_unstable_entry(self):
        bad = cfg(2 * 4096**3, 3 * 4096**2 * 2, 2.9e12, stable=False, n=4096)
        table = rr.matmul_table(matmul_result(**{"fp16-4096": bad}))
        fp16 = next(x for x in table.splitlines() if x.startswith("| fp16"))
        assert "not citable" in fp16 and "2.9" not in fp16

    def test_decode_table_lists_every_batch_and_hides_unstable_ones(self):
        rows = sweep()
        rows["summary"]["batch-8"]["stable"] = False
        a = rr.analyze(bandwidth_result(80.0), matmul_result(), rows)
        lines = rr.decode_table(a, rows).splitlines()
        assert len(lines) == 2 + 6
        assert rr.NOT_CITABLE in next(x for x in lines if x.startswith("| 8 |"))
        assert "memory" in next(x for x in lines if x.startswith("| 1 |"))

    def test_prediction_table_says_held_failed_or_not_evaluated(self):
        a = rr.analyze(bandwidth_result(40.0), matmul_result(), sweep())
        a["verdicts"]["skinny"] = None
        text = rr.predictions_table(a)
        assert "did not hold" in text and "not evaluated" in text and "| held |" in text

    def test_roofs_table_uses_the_cpu_roof_from_day_two(self):
        a = rr.analyze(bandwidth_result(80.0), matmul_result(), sweep())
        text = rr.roofs_table(a, {"peak_flops": 1.65e12, "bandwidth": 85.1e9})
        assert "| CPU, single precision | 1.65 | 85.1 | 19.4 |" in text
        assert "| GPU, half precision | 3.00 | 80.0 | 37.5 |" in text


class TestChart:
    def roofs(self):
        return [
            {"name": "Big GPU", "peak_flops": 1e15, "bandwidth": 3e12},
            {"name": "This chip, measured", "peak_flops": 3e12, "bandwidth": 8e10, "bold": True},
            {"name": "CPU", "peak_flops": 1.6e12, "bandwidth": 8.5e10, "dashed": True},
        ]

    def points(self):
        a = rr.analyze(bandwidth_result(80.0), matmul_result(), sweep())
        return a["points"]

    def test_the_axes_map_the_ends_of_their_ranges_to_the_ends_of_the_plot(self):
        assert rr.log_position(1.0, 1.0, 100.0, 0.0, 200.0) == pytest.approx(0.0)
        assert rr.log_position(100.0, 1.0, 100.0, 0.0, 200.0) == pytest.approx(200.0)
        assert rr.log_position(10.0, 1.0, 100.0, 0.0, 200.0) == pytest.approx(100.0)

    def test_a_roof_runs_up_the_slope_then_flat_across_the_top(self):
        (x0, y0), (x1, y1), (x2, y2) = rr.roof_polyline(1e12, 1e11)  # ridge at intensity 10
        assert y1 == pytest.approx(y2) and x2 > x1 > x0 and y0 > y1  # larger y is lower on screen
        assert x1 == pytest.approx(rr._px(10.0))

    def test_a_roof_whose_slope_starts_below_the_chart_is_clipped_to_the_bottom_edge(self):
        (_, y0), *_ = rr.roof_polyline(1e15, 1e9)  # at the left edge it would be 5e7, below 1e10
        assert y0 == pytest.approx(rr._py(rr.Y_RANGE[0]))

    def test_the_svg_is_well_formed_and_has_a_roof_and_a_marker_for_each_entry(self):
        points = self.points()
        svg = rr.roofline_svg(self.roofs(), points)
        root = ET.fromstring(svg)
        ns = {"s": "http://www.w3.org/2000/svg"}
        assert len(root.findall(".//s:polyline", ns)) == 3
        data_markers = [
            c for c in root.findall(".//s:circle", ns) if c.find("s:title", ns) is not None
        ]
        assert len(data_markers) == len(points)
        text = " ".join(t.text or "" for t in root.findall(".//s:text", ns))
        assert "Big GPU" in text and "FLOP per byte" in text and "measured: decode batch" in text

    def test_a_marker_sits_where_its_numbers_say(self):
        p = {"series": "stream", "key": "triad", "intensity": 1.0, "achieved_flops": 1e12}
        root = ET.fromstring(rr.roofline_svg([], [p]))
        ns = {"s": "http://www.w3.org/2000/svg"}
        marker = next(
            c for c in root.findall(".//s:circle", ns) if c.find("s:title", ns) is not None
        )
        assert float(marker.get("cx")) == pytest.approx(rr._px(1.0), abs=0.06)
        assert float(marker.get("cy")) == pytest.approx(rr._py(1e12), abs=0.06)

    def test_names_are_escaped(self):
        svg = rr.roofline_svg([{"name": "A & B <C>", "peak_flops": 1e12, "bandwidth": 1e11}], [])
        ET.fromstring(svg)  # raises if the name broke the markup
        assert "A &amp; B &lt;C&gt;" in svg
