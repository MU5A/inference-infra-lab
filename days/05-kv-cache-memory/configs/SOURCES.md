# Model configuration files for Day 5

Fetched on 2026-11-24 from huggingface.co, before the Day 5 run, with the owner's approval, so the day needs no
download. Each file is the repository's `config.json` at the pinned commit, byte for byte. The
machine-readable record, with the full URL and sha256 of each, is `manifest.json`, and a test fails if a file
changes.

| File | Repository | Commit | License | Bytes | What it covers |
|---|---|---|---|---|---|
| `Qwen2.5-0.5B-Instruct.config.json` | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | `7ae557604adf` | apache-2.0 | 659 | grouped-query attention, 14 query heads and 2 KV heads; the model family of the local GGUF |
| `Qwen2.5-7B-Instruct.config.json` | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | `a09a35458c70` | apache-2.0 | 663 | grouped-query attention at 7B, 28 query heads and 4 KV heads |
| `Mixtral-8x7B-v0.1.config.json` | [mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) | `fc7ac94680e3` | apache-2.0 | 720 | grouped-query attention with a mixture of experts, 8 experts and 2 per token |
| `OLMoE-1B-7B-0924.config.json` | [allenai/OLMoE-1B-7B-0924](https://huggingface.co/allenai/OLMoE-1B-7B-0924) | `6d84c48581ec` | apache-2.0 | 759 | mixture of experts with full multi-head attention, 64 experts and 8 per token |
| `DeepSeek-V2-Lite.config.json` | [deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) | `604d5664dddd` | other (license name `deepseek`) | 1522 | multi-head latent attention (`kv_lora_rank`) with a mixture of experts, 64 routed and 2 shared |

The files are facts about each model's shape and carry no weights. The DeepSeek-V2-Lite repository declares `license: other` with the name `deepseek`, not Apache-2.0; its
license file is https://github.com/deepseek-ai/DeepSeek-V2/blob/main/LICENSE-MODEL, to be read before
reusing anything beyond this configuration.
