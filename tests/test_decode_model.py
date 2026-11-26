"""The bound model, the summaries and the checks that judge Day 6's predictions."""

from __future__ import annotations

import pytest

from inferlab import decode_model as dm
from inferlab.gguf import from_summary

META = {
    "general.architecture": "qwen2",
    "qwen2.block_count": 24,
    "qwen2.attention.head_count": 14,
    "qwen2.attention.head_count_kv": 2,
    "qwen2.embedding_length": 896,
}


def header(names_and_sizes: dict[str, int], meta: dict | None = None):
    """A header whose tensors are float32 vectors of the given byte sizes."""
    tensors, offset = [], 0
    for name, size in names_and_sizes.items():
        tensors.append({"name": name, "dims": [size // 4], "type_id": 0, "offset": offset})
        offset += (size + 31) // 32 * 32
    return from_summary(
        {
            "version": 3,
            "alignment": 32,
            "data_offset": 64,
            "file_size": 64 + offset,
            "metadata": META if meta is None else meta,
            "tensors": tensors,
        }
    )


WITH_OUTPUT = {"token_embd.weight": 640, "output.weight": 960, "blk.0.w": 320}


def test_kv_bytes_per_token_match_day_five_s_hand_figure():
    assert dm.kv_bytes_per_token(header(WITH_OUTPUT)) == 24 * 2 * 128 * 2  # 12,288


def test_the_input_embedding_is_not_read_when_the_output_matrix_is_separate():
    assert dm.weight_bytes_per_token(header(WITH_OUTPUT)) == 960 + 320


def test_a_tied_embedding_is_also_the_output_so_it_is_read():
    tied = {"token_embd.weight": 640, "blk.0.w": 320}
    assert dm.weight_bytes_per_token(header(tied)) == 960


def test_bytes_grow_with_the_context_by_the_kv_bytes_and_the_bound_falls():
    g = header(WITH_OUTPUT)
    assert dm.bytes_per_token(g, 100) - dm.bytes_per_token(g, 0) == 100 * 12288
    assert dm.decode_bound(g, 1e9, 0) == pytest.approx(1e9 / 1280)
    assert dm.decode_bound(g, 1e9, 4096) < dm.decode_bound(g, 1e9, 0)


def test_the_context_during_a_decode_run_is_the_depth_plus_half_the_tokens_made():
    assert dm.mid_context(0, 128) == 64 and dm.mid_context(4096, 128) == 4160


@pytest.mark.parametrize(
    "meta",
    [
        META | {"general.architecture": "llama"},
        {k: v for k, v in META.items() if k != "qwen2.block_count"},
        META | {"qwen2.attention.head_count": True},
        META | {"qwen2.attention.head_count": 0},
        META | {"qwen2.embedding_length": 897},
    ],
)
def test_metadata_that_cannot_be_trusted_is_refused(meta):
    with pytest.raises(ValueError):
        dm.kv_bytes_per_token(header(WITH_OUTPUT, meta))


def test_a_file_without_an_embedding_or_a_nonsense_context_or_bandwidth_is_refused():
    with pytest.raises(ValueError):
        dm.weight_bytes_per_token(header({"output.weight": 64}))
    with pytest.raises(ValueError):
        dm.bytes_per_token(header(WITH_OUTPUT), -1)
    with pytest.raises(ValueError):
        dm.decode_bound(header(WITH_OUTPUT), 0, 10)


def test_a_rate_is_stable_only_with_three_runs_that_agree():
    assert dm.summarize_rate([100, 102, 98])["stable"]
    assert dm.summarize_rate([100, 102, 98])["median"] == 100
    assert not dm.summarize_rate([100, 100, 140])["stable"]
    assert not dm.summarize_rate([100, 100])["stable"]
    assert not dm.summarize_rate([100])["stable"]


def entry(prompt, gen, depth, samples):
    return {"n_prompt": prompt, "n_gen": gen, "n_depth": depth, "samples_ts": samples}


def test_one_llama_bench_output_gives_a_median_rate_per_test_and_depth():
    entries = [
        entry(512, 0, 0, [3000, 3200, 3100]),
        entry(0, 128, 0, [150, 160, 155]),
        entry(512, 0, 4096, [2900]),
        entry(0, 128, 4096, [140]),
    ]
    rates = dm.bench_rates(entries, (0, 4096))
    assert rates == {"pp@0": 3100, "tg@0": 155, "pp@4096": 2900, "tg@4096": 140}


def test_a_llama_bench_output_that_lacks_a_test_or_has_no_samples_is_refused():
    with pytest.raises(ValueError, match="lacks"):
        dm.bench_rates([entry(512, 0, 0, [1])], (0,))
    with pytest.raises(ValueError, match="no samples"):
        dm.bench_rates([entry(512, 0, 0, [])], (0,))


def test_ttft_slopes_are_per_extra_token_in_prompt_order():
    slopes = dm.ttft_slopes([(257, 0.100), (40, 0.030), (1469, 0.500)])
    assert slopes[0] == pytest.approx(0.070 / 217)
    assert slopes[1] == pytest.approx(0.400 / 1212)
    for bad in ([(40, 1.0)], [(40, 1.0), (40, 2.0)]):
        with pytest.raises(ValueError):
            dm.ttft_slopes(bad)


def rate(value, stable=True):
    return {"median": value, "stable": stable}


def results(pp, tg):
    return {m: {"pp@0": rate(pp[m]), "tg@0": rate(tg[m])} for m in dm.MODELS}


PP = {dm.SMALL_Q4: 3200, dm.SMALL_Q8: 3000, dm.SMALL_F16: 2800, dm.LARGE_Q4: 1000}
TG = {dm.SMALL_Q4: 160, dm.SMALL_Q8: 110, dm.SMALL_F16: 60, dm.LARGE_Q4: 70}


def test_bound_and_efficiency_checks():
    good = {m: 0.7 for m in dm.MODELS}
    assert dm.check_bound(good) == [] and dm.check_efficiency(good) == []
    assert dm.check_bound(good | {dm.SMALL_Q8: 1.0})
    assert dm.check_efficiency(good | {dm.SMALL_Q8: 0.5})
    assert dm.check_efficiency(good | {dm.SMALL_Q8: 0.96})


def test_the_larger_model_must_reach_strictly_more_of_its_bound():
    base = {dm.SMALL_Q4: 0.7, dm.LARGE_Q4: 0.8}
    assert dm.check_larger_model(base) == []
    assert dm.check_larger_model(base | {dm.LARGE_Q4: 0.7})


def test_prefill_compute_bound_needs_flat_prefill_and_a_large_decode_gap():
    assert dm.check_prefill_compute_bound(results(PP, TG)) == []
    spread = dm.check_prefill_compute_bound(results(PP | {dm.SMALL_F16: 2000}, TG))
    assert len(spread) == 1 and "prefill" in spread[0]
    close = dm.check_prefill_compute_bound(results(PP, TG | {dm.SMALL_F16: 100}))
    assert len(close) == 1 and "decodes" in close[0]


def test_prefill_of_the_bigger_model_is_checked_against_the_range():
    assert dm.check_prefill_size(results(PP, TG)) == []
    assert dm.check_prefill_size(results(PP | {dm.LARGE_Q4: 1600}, TG))
    assert dm.check_prefill_size(results(PP | {dm.LARGE_Q4: 700}, TG))


def test_depth_ratio_is_judged_with_an_absolute_tolerance():
    measured = {m: 0.90 for m in dm.MODELS}
    assert dm.check_depth(measured, {m: 0.85 for m in dm.MODELS}) == []
    assert len(dm.check_depth(measured, {m: 0.80 for m in dm.MODELS})) == 4


def test_ttft_must_be_linear_between_the_two_steps():
    linear = {"m": [(40, 0.03), (257, 0.10), (1469, 0.49)]}
    assert dm.check_ttft(linear) == []
    bent = {"m": [(40, 0.03), (257, 0.10), (1469, 0.80)]}
    assert dm.check_ttft(bent)
    assert dm.check_ttft({"m": [(40, 0.10), (257, 0.10), (1469, 0.49)]})


def test_a_prediction_nobody_judged_is_not_held():
    rows = dm.verdicts({"1": []})
    assert rows[0]["held"] and [r["number"] for r in rows] == [str(i) for i in range(1, 11)]
    assert all(r["failures"] == ["not judged"] and not r["held"] for r in rows[1:])


def test_files_that_read_the_same_bytes_are_expected_to_decode_at_the_same_speed():
    assert dm.check_same_bytes(results(PP, TG | {dm.LARGE_Q4: 60, dm.SMALL_F16: 60})) == []
    assert dm.check_same_bytes(results(PP, TG | {dm.LARGE_Q4: 74, dm.SMALL_F16: 60})) == []
    assert dm.check_same_bytes(results(PP, TG | {dm.LARGE_Q4: 76, dm.SMALL_F16: 60}))
    assert dm.check_same_bytes(results(PP, TG | {dm.LARGE_Q4: 44, dm.SMALL_F16: 60}))
