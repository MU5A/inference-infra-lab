# Day 5 log: KV cache and memory calculator

Appended as the work goes. Times are local (WAT).

## 0. Inputs fetched before the run, 2026-11-24 (not by the scheduled run)

The model configuration files this day validates against were fetched in an interactive session on
2026-11-24, with the owner's approval, because an unattended run may not download anything. They
are five public, ungated `config.json` files, each pinned to a commit, 4,323 bytes in total.
Sources, commits, licenses and sha256 are in `configs/SOURCES.md` and `configs/manifest.json`, and
`tests/test_day5_configs.py` fails if any of them changes. They cover grouped-query attention
(Qwen2.5 0.5B and 7B, Mixtral), mixture of experts (Mixtral, OLMoE, DeepSeek-V2-Lite) and
multi-head latent attention (DeepSeek-V2-Lite).

The comparison with what llama.cpp reports can use the local `models/qwen2.5-0.5b-instruct-q4_k_m.gguf`
and the llama.cpp build already here, so that part needs no download either. Anything else this day
needs that is not already on disk is still a stop (DAILY.md, "No downloads").
