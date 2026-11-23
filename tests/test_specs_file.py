"""The committed specification file: every figure traceable, every derivation re-computed."""

from __future__ import annotations

import pathlib

import pytest

from inferlab import roofline as rl

SPECS = (
    pathlib.Path(__file__).resolve().parents[1]
    / "days"
    / "03-roofline"
    / "specs"
    / "hardware_specs.json"
)


@pytest.fixture(scope="module")
def specs():
    return rl.load_specs(SPECS)  # raises if any figure lacks a source, date, or derivation


def device(specs, device_id):
    return next(d for d in specs["devices"] if d["id"] == device_id)


def test_the_file_validates_and_is_labelled_as_study(specs):
    assert specs["label"] == "L" and rl.validate_specs(specs) == []


def test_every_device_has_a_bandwidth_and_each_source_is_an_https_page(specs):
    for d in specs["devices"]:
        assert d["memory_bandwidth"]["source"].startswith("https://")
        for fig in d["compute"].values():
            assert fig["source"].startswith("https://")


def test_nvidia_hopper_dense_figures_are_half_of_the_printed_sparse_ones(specs):
    for device_id in ("h100_sxm", "h200_sxm"):
        fig = device(specs, device_id)["compute"]["fp16_dense"]
        assert fig["how"] == "derived" and "1,979" in fig["from"]
        assert fig["value"] == pytest.approx(1979 / 2)


def test_the_a100_dense_figure_is_the_printed_unstarred_value(specs):
    fig = device(specs, "a100_80gb_sxm")["compute"]["fp16_dense"]
    assert fig["how"] == "printed" and fig["value"] == 312 and "624 TFLOPS*" in fig["as_printed"]


def test_the_b200_figure_is_the_baseboard_sparse_value_halved_and_divided_by_eight(specs):
    fig = device(specs, "b200_sxm")["compute"]["fp16_dense"]
    assert fig["from"].startswith("HGX B200 8-GPU baseboard FP16/BF16 Tensor Core 36 PFLOPS")
    assert fig["value"] == pytest.approx(36 * 1000 / 2 / 8)


def test_the_laptop_chip_has_a_printed_bandwidth_and_no_invented_compute(specs):
    m3 = device(specs, "apple_m3")
    assert m3["memory_bandwidth"]["value"] == 100 and m3["memory_bandwidth"]["unit"] == "GB/s"
    assert m3["compute"] == {}
    assert rl.spec_roof(m3, "fp16_dense") is None


@pytest.mark.parametrize(
    "device_id, ridge",
    [
        ("a100_80gb_sxm", 312e12 / 2039e9),  # 153.0
        ("h100_sxm", 989.5e12 / 3.35e12),  # 295.4
        ("h200_sxm", 989.5e12 / 4.8e12),  # 206.1
        ("b200_sxm", 2250e12 / 8e12),  # 281.3
        ("mi300x", 1307.4e12 / 5.3e12),  # 246.7
    ],
)
def test_ridge_points_follow_from_the_printed_figures(specs, device_id, ridge):
    assert rl.spec_roof(device(specs, device_id), "fp16_dense").ridge == pytest.approx(ridge)
