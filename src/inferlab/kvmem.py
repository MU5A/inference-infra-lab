"""Day 5: a calculator for the memory an LLM needs, from its Hugging Face `config.json`.

Two things take memory while a model serves requests: the weights (fixed) and the KV cache (grows
with every token of every sequence in flight). This module computes both from the architecture,
so a model that is not on this laptop can still be sized.

    weights   = parameters x bytes per weight (set by the quantization)
    KV cache  = tokens x sequences x layers x (K elements + V elements per token) x bytes each

What it models, and what it refuses
-----------------------------------
Four architectures, each checked against a real configuration file and a published parameter
count (see `days/05-kv-cache-memory`): `qwen2` (grouped-query attention, dense), `mixtral` and
`olmoe` (mixture of experts) and `deepseek_v2` (multi-head latent attention plus mixture of
experts). Anything else raises `UnsupportedConfig` instead of returning a number that was not
checked, and so does a configuration whose meaning is ambiguous (sliding-window attention caps the
cache, which this module does not model; a missing `tie_word_embeddings`). When a value has to be
assumed, the assumption is recorded in `Model.notes`, never applied silently.

Not modelled: activation and compute buffers, the runtime's own overhead, memory fragmentation,
the CUDA or Metal context, and the cost of a draft model. A real process uses more than the sum
of weights and cache. Day 5's measurement shows how much more for one model on one engine.

Predictions, written BEFORE any run of llama.cpp for this day, before my GGUF reader touched the
real model file and before the published figures were looked up. The arithmetic for predictions
8 to 10 was done by hand while designing the module, so those three are not blind. The first seven
are about numbers I had not seen.

 M  (llama.cpp cbb1c1c, Qwen2.5-0.5B-Instruct Q4_K_M, this machine)
 1. The calculator's parameter count for the Qwen2.5-0.5B configuration equals, exactly, the number
    of elements in the GGUF file's tensors, and rounds to the `model params` figure llama.cpp
    prints, at its printed precision.
 2. The tensors the calculator expects (names and shapes) are exactly the tensors in the GGUF file:
    none missing, none extra, no shape different. In particular there is no separate output
    matrix, because the model ties its embeddings.
 3. Adding up every tensor's size from the quantization block table, each padded to the file's
    alignment, gives the size of the file's tensor-data region within 0.01%.
 4. For contexts that are multiples of 256 (512, 4096, 32768) and cache types f16, q8_0 and q4_0,
    the KV cache size llama.cpp reports is within 0.5% of the calculator's.
 5. A request for 1000 tokens of context is reported as 1024 cells (rounded up to a multiple of
    256), at a size within 0.5% of the calculator's 1024-cell figure and at least 2% above the
    calculator's 1000-token figure.
 6. The KV size llama.cpp reports is identical across the three launches of every configuration.
 7. In the memory breakdown printed at shutdown, the context column is within 2 MiB of the KV
    figure at every configuration, and the model columns, added across rows, are within 2% of the
    tensor-data region.

 S / L  (the calculator against published figures, looked up after this code was committed)
 8. The total parameters of all five committed configurations are within the rounding of the
    published figure (half a unit of its last printed digit).
 9. The activated parameters of Mixtral, OLMoE and DeepSeek-V2-Lite are within the same rounding
    of the published figure, where active means total minus the routed experts that are not
    selected, embeddings counted. This is the likeliest to fail: papers differ in what "activated"
    counts, and I did not look up which convention each one uses.
10. DeepSeek-V2-Lite's compressed cache is between 8 and 10 times smaller than the expanded
    per-head cache of the same model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

MIB = 1024**2
GIB = 1024**3

# (bytes per block, elements per block) of the cache types llama.cpp can store. Only f16, q8_0
# and q4_0 were checked against llama.cpp's own reports on Day 5.
CACHE_TYPES = {
    "f32": (4, 1),
    "f16": (2, 1),
    "bf16": (2, 1),
    "q8_0": (34, 32),
    "q4_0": (18, 32),
}

SUPPORTED = ("qwen2", "mixtral", "olmoe", "deepseek_v2")
MLA_MODES = ("compressed", "expanded")


class UnsupportedConfig(ValueError):
    """The configuration is outside what this module was checked on, or is ambiguous."""


@dataclass(frozen=True)
class Model:
    name: str
    model_type: str
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    v_head_dim: int
    vocab: int
    tied: bool
    qkv_bias: bool
    o_bias: bool
    qk_norm: bool
    ffn: int  # the dense MLP width
    experts: int = 0
    experts_per_token: int = 0
    expert_ffn: int = 0
    shared_experts: int = 0
    first_dense_layers: int = 0
    moe_layer_freq: int = 1
    mla: bool = False
    kv_lora_rank: int = 0
    q_lora_rank: int | None = None
    qk_nope: int = 0
    qk_rope: int = 0
    notes: tuple[str, ...] = ()

    def is_moe_layer(self, index: int) -> bool:
        return (
            self.experts > 0
            and index >= self.first_dense_layers
            and index % self.moe_layer_freq == 0
        )


def _int(cfg: Mapping, key: str, *, minimum: int = 1) -> int:
    value = cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise UnsupportedConfig(f"{key} must be an integer, got {value!r}")
    if value < minimum:
        raise UnsupportedConfig(f"{key} must be at least {minimum}, got {value}")
    return value


def load_model(cfg: Mapping, name: str = "model") -> Model:
    """Read a Hugging Face configuration. Raises UnsupportedConfig rather than guess."""
    model_type = cfg.get("model_type")
    if model_type not in SUPPORTED:
        raise UnsupportedConfig(
            f"model_type {model_type!r} was not checked; supported: {', '.join(SUPPORTED)}"
        )
    sliding = cfg.get("sliding_window")
    if cfg.get("use_sliding_window") is True or (
        model_type in ("mistral", "mixtral") and sliding is not None
    ):
        raise UnsupportedConfig("sliding-window attention caps the cache and is not modelled")
    if model_type == "qwen2" and sliding is not None and "use_sliding_window" not in cfg:
        raise UnsupportedConfig("sliding_window is set and use_sliding_window is missing")
    if "tie_word_embeddings" not in cfg:
        raise UnsupportedConfig("tie_word_embeddings is missing, and the default differs by class")
    if cfg.get("mlp_bias"):
        raise UnsupportedConfig("a bias on the MLP is not modelled")
    if cfg.get("mlp_only_layers") or cfg.get("decoder_sparse_step", 1) != 1:
        raise UnsupportedConfig(
            "a mix of dense and sparse layers is not modelled, apart from deepseek_v2's "
            "leading dense layers"
        )

    hidden = _int(cfg, "hidden_size")
    heads = _int(cfg, "num_attention_heads")
    notes = []
    if cfg.get("num_key_value_heads") is None:
        kv_heads = heads
        notes.append("num_key_value_heads absent: assumed equal to the attention heads (MHA)")
    else:
        kv_heads = _int(cfg, "num_key_value_heads")
    if heads % kv_heads:
        raise UnsupportedConfig(f"{heads} attention heads do not divide into {kv_heads} KV heads")
    if cfg.get("head_dim") is not None:
        head_dim = _int(cfg, "head_dim")
    elif hidden % heads:
        raise UnsupportedConfig(f"hidden_size {hidden} is not divisible by {heads} heads")
    else:
        head_dim = hidden // heads

    common = {
        "name": name,
        "model_type": model_type,
        "hidden": hidden,
        "layers": _int(cfg, "num_hidden_layers"),
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "v_head_dim": head_dim,
        "vocab": _int(cfg, "vocab_size"),
        "tied": bool(cfg["tie_word_embeddings"]),
        "ffn": _int(cfg, "intermediate_size"),
        "notes": tuple(notes),
    }
    if model_type == "qwen2":
        return Model(**common, qkv_bias=True, o_bias=False, qk_norm=False)
    if model_type == "mixtral":
        return Model(
            **common,
            qkv_bias=False,
            o_bias=False,
            qk_norm=False,
            experts=_int(cfg, "num_local_experts"),
            experts_per_token=_int(cfg, "num_experts_per_tok"),
            expert_ffn=common["ffn"],
        )
    if model_type == "olmoe":
        bias = bool(cfg.get("attention_bias", False))
        return Model(
            **common,
            qkv_bias=bias,
            o_bias=bias,
            qk_norm=True,
            experts=_int(cfg, "num_experts"),
            experts_per_token=_int(cfg, "num_experts_per_tok"),
            expert_ffn=common["ffn"],
        )
    if cfg.get("attention_bias"):
        raise UnsupportedConfig("a bias on the MLA projections is not modelled")
    nope, rope = _int(cfg, "qk_nope_head_dim"), _int(cfg, "qk_rope_head_dim")
    q_rank = cfg.get("q_lora_rank")
    return Model(
        **(common | {"head_dim": nope + rope, "v_head_dim": _int(cfg, "v_head_dim")}),
        qkv_bias=False,
        o_bias=False,
        qk_norm=False,
        experts=_int(cfg, "n_routed_experts"),
        experts_per_token=_int(cfg, "num_experts_per_tok"),
        expert_ffn=_int(cfg, "moe_intermediate_size"),
        shared_experts=_int(cfg, "n_shared_experts", minimum=0),
        first_dense_layers=_int(cfg, "first_k_dense_replace", minimum=0),
        moe_layer_freq=_int(cfg, "moe_layer_freq"),
        mla=True,
        kv_lora_rank=_int(cfg, "kv_lora_rank"),
        q_lora_rank=None if q_rank is None else _int(cfg, "q_lora_rank"),
        qk_nope=nope,
        qk_rope=rope,
    )


def load_config_file(path: str | Path) -> Model:
    path = Path(path)
    try:
        cfg = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise UnsupportedConfig(f"cannot read {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise UnsupportedConfig(f"{path} is not a JSON object")
    return load_model(cfg, name=path.name.removesuffix(".config.json").removesuffix(".json"))


# ---------------------------------------------------------------- parameters


@dataclass(frozen=True)
class Params:
    total: int
    active: int
    input_embedding: int
    lm_head: int  # 0 when tied to the input embedding

    @property
    def non_embedding(self) -> int:
        return self.total - self.input_embedding - self.lm_head


def attention_params(m: Model) -> int:
    if m.mla:
        q_out = m.heads * (m.qk_nope + m.qk_rope)
        if m.q_lora_rank is None:
            q = m.hidden * q_out
        else:
            q = m.hidden * m.q_lora_rank + m.q_lora_rank + m.q_lora_rank * q_out
        kv_a = m.hidden * (m.kv_lora_rank + m.qk_rope) + m.kv_lora_rank
        kv_b = m.kv_lora_rank * m.heads * (m.qk_nope + m.v_head_dim)
        return q + kv_a + kv_b + m.heads * m.v_head_dim * m.hidden
    q_out, k_out, v_out = m.heads * m.head_dim, m.kv_heads * m.head_dim, m.kv_heads * m.v_head_dim
    total = m.hidden * (q_out + k_out + v_out) + m.heads * m.v_head_dim * m.hidden
    if m.qkv_bias:
        total += q_out + k_out + v_out
    if m.o_bias:
        total += m.hidden
    if m.qk_norm:
        total += q_out + k_out
    return total


def _mlp(m: Model, index: int) -> tuple[int, int]:
    """(total, active) parameters of one layer's feed-forward block."""
    if not m.is_moe_layer(index):
        dense = 3 * m.hidden * m.ffn
        return dense, dense
    expert = 3 * m.hidden * m.expert_ffn
    shared = 3 * m.hidden * m.expert_ffn * m.shared_experts
    router = m.hidden * m.experts
    return m.experts * expert + shared + router, m.experts_per_token * expert + shared + router


def param_counts(m: Model) -> Params:
    attention = attention_params(m)
    norms = 2 * m.hidden
    total = active = m.hidden  # the final norm
    for i in range(m.layers):
        mlp_total, mlp_active = _mlp(m, i)
        total += attention + norms + mlp_total
        active += attention + norms + mlp_active
    embedding = m.vocab * m.hidden
    head = 0 if m.tied else embedding
    return Params(
        total=total + embedding + head,
        active=active + embedding + head,
        input_embedding=embedding,
        lm_head=head,
    )


def weight_bytes(parameters: int, bits_per_weight: float) -> int:
    if parameters < 0 or not bits_per_weight > 0:
        raise ValueError("parameters must not be negative and bits per weight must be positive")
    return math.ceil(parameters * bits_per_weight / 8)


# ------------------------------------------------------------------ KV cache


def kv_elements(m: Model, mla: str = "compressed") -> tuple[int, int]:
    """K and V elements stored per token per layer."""
    if mla not in MLA_MODES:
        raise ValueError(f"mla must be one of {MLA_MODES}")
    if not m.mla:
        return m.kv_heads * m.head_dim, m.kv_heads * m.v_head_dim
    if mla == "compressed":
        return m.kv_lora_rank + m.qk_rope, 0
    return m.heads * (m.qk_nope + m.qk_rope), m.heads * m.v_head_dim


def _row_bytes(elements: int, cache_type: str) -> int:
    if cache_type not in CACHE_TYPES:
        raise ValueError(f"cache type must be one of {sorted(CACHE_TYPES)}, got {cache_type!r}")
    block_bytes, block_elements = CACHE_TYPES[cache_type]
    if elements % block_elements:
        raise UnsupportedConfig(
            f"{elements} elements per token per layer is not a whole number of "
            f"{cache_type} blocks of {block_elements} (from ggml's block layout, not measured)"
        )
    return elements // block_elements * block_bytes


def kv_bytes_per_token(
    m: Model,
    cache_type: str = "f16",
    *,
    v_cache_type: str | None = None,
    mla: str = "compressed",
) -> int:
    k, v = kv_elements(m, mla)
    per_layer = _row_bytes(k, cache_type)
    if v:
        per_layer += _row_bytes(v, v_cache_type or cache_type)
    return m.layers * per_layer


def kv_cache_bytes(
    m: Model,
    cells: int,
    *,
    sequences: int = 1,
    cache_type: str = "f16",
    v_cache_type: str | None = None,
    mla: str = "compressed",
) -> int:
    if cells < 0 or sequences < 1:
        raise ValueError("cells must not be negative and sequences must be at least 1")
    return cells * sequences * kv_bytes_per_token(m, cache_type, v_cache_type=v_cache_type, mla=mla)


# ------------------------------------------------------------ whole estimate


@dataclass(frozen=True)
class Estimate:
    weights: int
    kv: int

    @property
    def total(self) -> int:
        return self.weights + self.kv


def estimate(
    m: Model,
    *,
    context: int,
    sequences: int = 1,
    cache_type: str = "f16",
    bits_per_weight: float = 16.0,
    mla: str = "compressed",
) -> Estimate:
    """Weights plus KV cache. Compute buffers and runtime overhead are NOT included."""
    return Estimate(
        weights=weight_bytes(param_counts(m).total, bits_per_weight),
        kv=kv_cache_bytes(m, context, sequences=sequences, cache_type=cache_type, mla=mla),
    )


def max_context(
    m: Model,
    budget_bytes: int,
    *,
    sequences: int = 1,
    cache_type: str = "f16",
    bits_per_weight: float = 16.0,
    mla: str = "compressed",
) -> int:
    """Tokens per sequence that fit in the budget after the weights; 0 if the weights do not."""
    if sequences < 1:
        raise ValueError("sequences must be at least 1")
    room = budget_bytes - weight_bytes(param_counts(m).total, bits_per_weight)
    per_token = kv_bytes_per_token(m, cache_type, mla=mla) * sequences
    return max(0, room // per_token)


# ----------------------------------------------------------------------- CLI


def describe(m: Model, args: argparse.Namespace) -> str:
    params = param_counts(m)
    est = estimate(
        m,
        context=args.context,
        sequences=args.sequences,
        cache_type=args.cache_type,
        bits_per_weight=args.bits_per_weight,
        mla=args.mla,
    )
    lines = [
        f"{m.name} ({m.model_type})",
        f"  parameters   {params.total / 1e9:.3f} B total, {params.active / 1e9:.3f} B active",
        f"  weights      {est.weights / GIB:.2f} GiB at {args.bits_per_weight} bits per weight",
        f"  KV per token {kv_bytes_per_token(m, args.cache_type, mla=args.mla) / 1024:.1f} KiB "
        f"({args.cache_type})",
        f"  KV cache     {est.kv / GIB:.2f} GiB for {args.sequences} x {args.context} tokens",
        f"  total        {est.total / GIB:.2f} GiB, before compute buffers and runtime overhead",
    ]
    if args.budget_gib is not None:
        fit = max_context(
            m,
            int(args.budget_gib * GIB),
            sequences=args.sequences,
            cache_type=args.cache_type,
            bits_per_weight=args.bits_per_weight,
            mla=args.mla,
        )
        lines.append(f"  in {args.budget_gib} GiB   {fit} tokens per sequence fit")
    lines += [f"  note: {n}" for n in m.notes]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m inferlab.kvmem",
        description="Weights and KV cache memory from a Hugging Face config.json.",
    )
    parser.add_argument("config", type=Path, help="a Hugging Face config.json")
    parser.add_argument("--context", type=int, default=4096, help="tokens per sequence")
    parser.add_argument("--sequences", type=int, default=1, help="sequences in flight")
    parser.add_argument("--cache-type", default="f16", choices=sorted(CACHE_TYPES))
    parser.add_argument("--bits-per-weight", type=float, default=16.0)
    parser.add_argument("--mla", default="compressed", choices=MLA_MODES)
    parser.add_argument("--budget-gib", type=float, help="also print the context that fits")
    args = parser.parse_args(argv)
    try:
        print(describe(load_config_file(args.config), args))
    except (UnsupportedConfig, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
