"""The Day 6 report on synthetic results: it must judge what repeats and nothing else."""

from __future__ import annotations

import pytest

from inferlab import decode_model as dm
from inferlab import decode_report

META = {
    "general.architecture": "qwen2",
    "qwen2.block_count": 2,
    "qwen2.attention.head_count": 4,
    "qwen2.attention.head_count_kv": 2,
    "qwen2.embedding_length": 64,
}
SIZES = {dm.SMALL_Q4: 1000, dm.SMALL_Q8: 1500, dm.SMALL_F16: 3000, dm.LARGE_Q4: 3000}
PP = {dm.SMALL_Q4: 3200, dm.SMALL_Q8: 3000, dm.SMALL_F16: 2800, dm.LARGE_Q4: 1000}
BANDWIDTH = 1e9


def summary(size_kib: int) -> dict:
    size = size_kib * 1024
    tensors = [
        {"name": "token_embd.weight", "dims": [16], "type_id": 0, "offset": 0},
        {"name": "output.weight", "dims": [size // 4], "type_id": 0, "offset": 64},
    ]
    return {
        "version": 3,
        "alignment": 32,
        "data_offset": 64,
        "file_size": 128 + size,
        "metadata": META,
        "tensors": tensors,
    }


def bound(key: str, context: float) -> float:
    return BANDWIDTH / (SIZES[key] * 1024 + context * 2 * 2 * 2 * 16 * 2)


def inputs(efficiency=0.7, noise=1.0, ttft_cv=0.01):
    summaries = {k: summary(v) for k, v in SIZES.items()}
    models = {}
    for key in dm.MODELS:
        eff = efficiency + (0.1 if key == dm.LARGE_Q4 else 0)
        runs = []
        for factor in (1.0, 1.0, noise):
            runs.append(
                {
                    "pp@0": PP[key] * factor,
                    "tg@0": eff * bound(key, 64) * factor,
                    "pp@4096": PP[key],
                    "tg@4096": eff * bound(key, 4160) * factor,
                }
            )
        models[key] = {"runs": runs}
    bench = {"depth": 4096, "n_prompt": 512, "n_gen": 128, "models": models}
    prompts = {
        name: {
            "prompt_tokens": [tokens],
            "ttft_s": {"median": 0.02 + tokens * 0.0003},
            "cv_ttft": ttft_cv,
            "failed": 0,
        }
        for name, tokens in (("short", 40), ("medium", 257), ("long", 1469))
    }
    ttft = {"models": {k: {"prompts": prompts} for k in dm.MODELS}}
    router = [
        {
            "shape": "s",
            "experts": 64,
            "distribution": d,
            "batch": b,
            "mean_imbalance": m,
            "min_imbalance": m,
            "max_imbalance": m,
            "trials": 1,
        }
        for d, b, m in (("uniform", 8, 3.9), ("uniform", 4096, 1.1), ("zipf", 4096, 6.0))
    ]
    return summaries, bench, ttft, router, BANDWIDTH


def held(analysis: dict) -> dict:
    return {v["number"]: v["held"] for v in analysis["verdicts"]}


def test_results_that_follow_the_model_hold_the_bound_predictions():
    a = decode_report.analyze(*inputs())
    assert a["models"][dm.SMALL_Q4]["efficiency"] == pytest.approx(0.7)
    row = a["models"][dm.SMALL_F16]
    assert row["depth_ratio_measured"] == pytest.approx(row["depth_ratio_predicted"])
    verdicts = held(a)
    assert all(verdicts[n] for n in ("1", "2", "3", "5", "6", "7", "8", "9"))


def test_a_measurement_that_did_not_repeat_is_shown_but_never_judged():
    a = decode_report.analyze(*inputs(noise=1.5))
    verdicts = {v["number"]: v for v in a["verdicts"]}
    for n in ("1", "2", "3", "4", "5", "6"):
        assert not verdicts[n]["held"]
        assert all(f.startswith("not judged") for f in verdicts[n]["failures"])
    assert "(unstable)" in decode_report.render(a)


def test_an_unrepeatable_first_token_time_blocks_only_that_prediction():
    a = decode_report.analyze(*inputs(ttft_cv=0.2))
    verdicts = {v["number"]: v for v in a["verdicts"]}
    assert not verdicts["7"]["held"] and verdicts["7"]["failures"][0].startswith("not judged")
    assert verdicts["1"]["held"]


def test_runs_that_disagree_on_the_prompt_length_are_refused():
    summaries, bench, ttft, router, bandwidth = inputs()
    ttft["models"][dm.SMALL_Q4]["prompts"]["short"]["prompt_tokens"] = [40, 41]
    with pytest.raises(ValueError, match="prompt length"):
        decode_report.analyze(summaries, bench, ttft, router, bandwidth)


def test_the_tables_name_every_model_and_prediction():
    text = decode_report.render(decode_report.analyze(*inputs()))
    for key in dm.MODELS:
        assert key in text
    for number in range(1, 10):
        assert f"\n| {number} | " in text
    assert "Bandwidth: 1.00 GB/s" in text
