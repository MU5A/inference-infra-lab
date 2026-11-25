"""The memory calculator: exact values worked out by hand, and configurations built to break it."""

from __future__ import annotations

import json
import pathlib

import pytest

from inferlab import kvmem
from inferlab.kvmem import UnsupportedConfig

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "days" / "05-kv-cache-memory" / "configs"

needs_configs = pytest.mark.skipif(not CONFIGS.exists(), reason="no Day 5 configs committed")


def model(name: str) -> kvmem.Model:
    return kvmem.load_config_file(CONFIGS / f"{name}.config.json")


# Worked out by hand from the architecture, not produced by running the code. The first hand figure
# for DeepSeek-V2-Lite was 800,000 too high (a slip adding 26 layers); the test caught it, and the
# corrected value was confirmed by listing all 5,291 of its tensors one by one, outside this code.
HAND_TOTAL = {
    "Qwen2.5-0.5B-Instruct": 494_032_768,
    "Qwen2.5-7B-Instruct": 7_615_616_512,
    "Mixtral-8x7B-v0.1": 46_702_792_704,
    "OLMoE-1B-7B-0924": 6_919_161_856,
    "DeepSeek-V2-Lite": 15_706_484_224,
}
HAND_ACTIVE = {
    "Mixtral-8x7B-v0.1": 12_879_925_248,
    "OLMoE-1B-7B-0924": 1_282_017_280,
    "DeepSeek-V2-Lite": 2_661_150_208,
}


@needs_configs
@pytest.mark.parametrize("name", sorted(HAND_TOTAL))
def test_total_parameters_match_the_hand_count(name):
    assert kvmem.param_counts(model(name)).total == HAND_TOTAL[name]


@needs_configs
@pytest.mark.parametrize("name", sorted(HAND_ACTIVE))
def test_active_parameters_match_the_hand_count(name):
    assert kvmem.param_counts(model(name)).active == HAND_ACTIVE[name]


@needs_configs
def test_a_dense_model_has_every_parameter_active():
    counts = kvmem.param_counts(model("Qwen2.5-7B-Instruct"))
    assert counts.active == counts.total


@needs_configs
def test_kv_bytes_per_token_match_the_hand_count_in_half_precision():
    assert kvmem.kv_bytes_per_token(model("Qwen2.5-0.5B-Instruct")) == 24 * 2 * 128 * 2
    assert kvmem.kv_bytes_per_token(model("Qwen2.5-7B-Instruct")) == 28 * 2 * 512 * 2
    assert kvmem.kv_bytes_per_token(model("Mixtral-8x7B-v0.1")) == 32 * 2 * 1024 * 2
    assert kvmem.kv_bytes_per_token(model("OLMoE-1B-7B-0924")) == 16 * 2 * 2048 * 2


@needs_configs
def test_latent_attention_stores_one_compressed_vector_not_keys_and_values():
    m = model("DeepSeek-V2-Lite")
    assert kvmem.kv_bytes_per_token(m, mla="compressed") == 27 * (512 + 64) * 2
    assert kvmem.kv_bytes_per_token(m, mla="expanded") == 27 * (16 * 192 + 16 * 128) * 2


@needs_configs
def test_the_expert_count_changes_the_weights_but_never_the_cache():
    """A mixture of experts keeps all its experts in memory, and its cache follows attention."""
    m = model("Mixtral-8x7B-v0.1")
    fewer = kvmem.Model(**(m.__dict__ | {"experts": 2}))
    assert kvmem.param_counts(fewer).total < kvmem.param_counts(m).total
    assert kvmem.kv_bytes_per_token(fewer) == kvmem.kv_bytes_per_token(m)


@needs_configs
def test_tied_embeddings_save_exactly_one_output_matrix():
    m = model("Qwen2.5-7B-Instruct")
    tied = kvmem.Model(**(m.__dict__ | {"tied": True}))
    saved = kvmem.param_counts(m).total - kvmem.param_counts(tied).total
    assert saved == m.vocab * m.hidden


@needs_configs
def test_quantized_cache_sizes_are_the_block_fractions_of_half_precision():
    m = model("Qwen2.5-0.5B-Instruct")
    f16 = kvmem.kv_bytes_per_token(m, "f16")
    assert kvmem.kv_bytes_per_token(m, "q8_0") * 32 == f16 * 17
    assert kvmem.kv_bytes_per_token(m, "q4_0") * 32 == f16 * 9


@needs_configs
def test_estimate_adds_weights_and_cache_and_scales_with_sequences():
    m = model("Qwen2.5-0.5B-Instruct")
    one = kvmem.estimate(m, context=1000, sequences=1, bits_per_weight=4.5)
    four = kvmem.estimate(m, context=1000, sequences=4, bits_per_weight=4.5)
    assert one.total == one.weights + one.kv
    assert four.kv == 4 * one.kv and four.weights == one.weights
    assert one.weights == kvmem.weight_bytes(HAND_TOTAL["Qwen2.5-0.5B-Instruct"], 4.5)


@needs_configs
def test_max_context_is_the_last_length_that_fits_and_zero_when_weights_do_not():
    m = model("Qwen2.5-0.5B-Instruct")
    budget = 2 * kvmem.GIB
    fit = kvmem.max_context(m, budget, bits_per_weight=16)
    assert kvmem.estimate(m, context=fit).total <= budget
    assert kvmem.estimate(m, context=fit + 1).total > budget
    assert kvmem.max_context(m, 1000, bits_per_weight=16) == 0


# ------------------------------------------------------------------ hostile


def good_qwen(**changes) -> dict:
    cfg = {
        "model_type": "qwen2",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "vocab_size": 100,
        "tie_word_embeddings": False,
        "use_sliding_window": False,
    }
    return cfg | changes


def test_the_hand_built_config_loads_so_the_failures_below_are_the_changes():
    kvmem.load_model(good_qwen())


@pytest.mark.parametrize(
    "changes",
    [
        {"model_type": "llama"},
        {"model_type": None},
        {"use_sliding_window": True},
        {"sliding_window": 4096, "use_sliding_window": None},
        {"num_key_value_heads": 3},
        {"hidden_size": 63},
        {"num_hidden_layers": 0},
        {"num_hidden_layers": True},
        {"num_hidden_layers": 2.0},
        {"vocab_size": -5},
        {"mlp_bias": True},
        {"decoder_sparse_step": 2},
        {"mlp_only_layers": [0]},
    ],
)
def test_a_config_outside_what_was_checked_is_refused(changes):
    cfg = good_qwen(**changes)
    for key in [k for k, v in changes.items() if v is None]:
        del cfg[key]
    with pytest.raises(UnsupportedConfig):
        kvmem.load_model(cfg)


def test_a_missing_tie_flag_is_refused_because_its_default_differs_by_class():
    cfg = good_qwen()
    del cfg["tie_word_embeddings"]
    with pytest.raises(UnsupportedConfig):
        kvmem.load_model(cfg)


def test_sliding_window_that_is_switched_off_does_not_cap_the_cache():
    m = kvmem.load_model(good_qwen(sliding_window=4096, use_sliding_window=False))
    assert kvmem.kv_bytes_per_token(m) == 2 * 2 * (2 * 16) * 2


def test_mixtral_with_a_window_is_refused():
    cfg = {
        "model_type": "mixtral",
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
        "vocab_size": 10,
        "tie_word_embeddings": False,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "sliding_window": 4096,
    }
    with pytest.raises(UnsupportedConfig):
        kvmem.load_model(cfg)
    assert kvmem.load_model(cfg | {"sliding_window": None})


def test_a_missing_kv_head_count_is_assumed_to_be_full_attention_and_says_so():
    cfg = good_qwen()
    del cfg["num_key_value_heads"]
    m = kvmem.load_model(cfg)
    assert m.kv_heads == m.heads
    assert any("MHA" in note for note in m.notes)


def test_an_explicit_head_dimension_overrides_hidden_over_heads():
    m = kvmem.load_model(good_qwen(head_dim=32))
    assert m.head_dim == 32
    assert kvmem.kv_elements(m) == (2 * 32, 2 * 32)


def test_a_cache_row_that_is_not_whole_blocks_is_refused_only_when_quantized():
    m = kvmem.load_model(good_qwen(head_dim=20))  # 2 heads x 20 = 40, not a multiple of 32
    assert kvmem.kv_bytes_per_token(m, "f16") > 0
    with pytest.raises(UnsupportedConfig):
        kvmem.kv_bytes_per_token(m, "q8_0")


@pytest.mark.parametrize("cache_type", ["f8", "", "F16"])
def test_an_unknown_cache_type_is_refused(cache_type):
    with pytest.raises(ValueError):
        kvmem.kv_bytes_per_token(kvmem.load_model(good_qwen()), cache_type)


@pytest.mark.parametrize("cells, sequences", [(-1, 1), (10, 0), (10, -2)])
def test_a_negative_length_or_no_sequence_is_refused(cells, sequences):
    with pytest.raises(ValueError):
        kvmem.kv_cache_bytes(kvmem.load_model(good_qwen()), cells, sequences=sequences)


@pytest.mark.parametrize("bits", [0, -4, float("nan")])
def test_nonsense_bits_per_weight_are_refused(bits):
    with pytest.raises(ValueError):
        kvmem.weight_bytes(1000, bits)


def test_the_zero_length_cache_is_zero_bytes():
    assert kvmem.kv_cache_bytes(kvmem.load_model(good_qwen()), 0) == 0


def test_a_moe_layer_pattern_follows_first_dense_layers_and_frequency():
    base = {
        "model_type": "deepseek_v2",
        "hidden_size": 64,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 128,
        "vocab_size": 50,
        "tie_word_embeddings": False,
        "kv_lora_rank": 32,
        "q_lora_rank": None,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 8,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 2,
    }
    m = kvmem.load_model(base)
    assert [m.is_moe_layer(i) for i in range(6)] == [False, False, True, False, True, False]
    no_dense = kvmem.load_model(base | {"first_k_dense_replace": 0, "moe_layer_freq": 1})
    assert all(no_dense.is_moe_layer(i) for i in range(6))


def test_q_compression_adds_the_two_low_rank_matrices_and_a_norm():
    base = {
        "model_type": "deepseek_v2",
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 128,
        "vocab_size": 50,
        "tie_word_embeddings": False,
        "kv_lora_rank": 32,
        "q_lora_rank": None,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 8,
        "n_routed_experts": 2,
        "n_shared_experts": 0,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 16,
        "first_k_dense_replace": 0,
        "moe_layer_freq": 1,
    }
    plain = kvmem.attention_params(kvmem.load_model(base))
    low_rank = kvmem.attention_params(kvmem.load_model(base | {"q_lora_rank": 24}))
    q_out = 4 * 16
    assert low_rank - plain == 64 * 24 + 24 + 24 * q_out - 64 * q_out


# ----------------------------------------------------------------------- CLI


def test_the_command_line_prints_the_numbers_and_the_assumptions(tmp_path, capsys):
    cfg = good_qwen()
    del cfg["num_key_value_heads"]
    path = tmp_path / "tiny.config.json"
    path.write_text(json.dumps(cfg))
    code = kvmem.main([str(path), "--context", "1000", "--budget-gib", "1"])
    out = capsys.readouterr().out
    assert code == 0
    assert "tiny (qwen2)" in out and "before compute buffers" in out and "assumed" in out
    assert "tokens per sequence fit" in out


@pytest.mark.parametrize("content", ["not json", "[1, 2]", json.dumps({"model_type": "llama"})])
def test_the_command_line_refuses_bad_files_with_a_message_and_status_2(tmp_path, capsys, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    assert kvmem.main([str(path)]) == 2
    assert "refused" in capsys.readouterr().err


def test_the_command_line_refuses_a_missing_file(tmp_path, capsys):
    assert kvmem.main([str(tmp_path / "nope.json")]) == 2
    assert "refused" in capsys.readouterr().err


def test_counting_the_output_matrix_separately_adds_exactly_one_embedding_matrix(tmp_path, capsys):
    cfg = good_qwen(tie_word_embeddings=True)
    path = tmp_path / "tied.config.json"
    path.write_text(json.dumps(cfg))
    kvmem.main([str(path), "--bits-per-weight", "8"])
    tied_out = capsys.readouterr().out
    kvmem.main([str(path), "--bits-per-weight", "8", "--output-matrix-separate"])
    separate_out = capsys.readouterr().out
    m = kvmem.load_model(cfg)
    tied_total = kvmem.param_counts(m).total
    assert f"{tied_total / 1e9:.3f} B total" in tied_out
    assert f"{(tied_total + m.vocab * m.hidden) / 1e9:.3f} B total" in separate_out
    assert "counted separately" in separate_out and "counted separately" not in tied_out


def test_the_separate_output_option_changes_nothing_for_a_model_that_already_has_one(
    tmp_path, capsys
):
    path = tmp_path / "untied.config.json"
    path.write_text(json.dumps(good_qwen(tie_word_embeddings=False)))
    kvmem.main([str(path)])
    plain = capsys.readouterr().out
    kvmem.main([str(path), "--output-matrix-separate"])
    assert capsys.readouterr().out == plain


def test_weight_bytes_round_up_to_a_whole_byte():
    assert kvmem.weight_bytes(10, 4.5) == 6  # 45 bits need 6 bytes
    assert kvmem.weight_bytes(8, 4) == 4
    assert kvmem.weight_bytes(0, 4) == 0
