### Decode against the memory-bandwidth bound

Bandwidth: 87.22 GB/s, Day 3's MLX GPU triad.

| Model file | File (GB) | Weights read per token (MB) | All bytes per token (MB) | Bound (tokens/s) | Measured decode (tokens/s) | Range over runs | Measured / bound |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b-q4_k_m | 0.49 | 391.9 | 392.6 | 222.1 | 159.8 | 138.7 to 166.2 | 0.72 |
| qwen2.5-0.5b-q8_0 | 0.68 | 525.1 | 525.9 | 165.9 | 143.8 | 140.8 to 144.4 | 0.87 |
| qwen2.5-0.5b-fp16 | 1.27 | 988.2 | 989.0 | 88.2 | 85.0 | 84.9 to 85.9 | 0.96 |
| qwen2.5-1.5b-q4_k_m | 1.12 | 980.1 | 981.9 | 88.8 | 80.0 | 78.8 to 80.4 | 0.90 |

### Prefill

| Model file | Parameters in file (M) | Prefill, 512 tokens (tokens/s) | Range over runs | Prefill / decode |
|---|---|---|---|---|
| qwen2.5-0.5b-q4_k_m | 630 | 3170 | 3129 to 3228 | 19.8 |
| qwen2.5-0.5b-q8_0 | 630 | 3398 | 3384 to 3451 | 23.6 |
| qwen2.5-0.5b-fp16 | 630 | 3590 | 3551 to 3592 | 42.2 |
| qwen2.5-1.5b-q4_k_m | 1777 | 985 | 985 to 986 | 12.3 |

### Decode deep in the context (after 4,096 tokens)

| Model file | KV per token (KiB) | Bytes per token at 4096 (MB) | Decode at 4096 (tokens/s) | Deep / start, measured | Deep / start, predicted |
|---|---|---|---|---|---|
| qwen2.5-0.5b-q4_k_m | 12 | 443.0 | 145.3 (unstable) | 0.910 | 0.886 |
| qwen2.5-0.5b-q8_0 | 12 | 576.2 | 129.2 | 0.898 | 0.913 |
| qwen2.5-0.5b-fp16 | 12 | 1039.3 | 81.4 | 0.957 | 0.952 |
| qwen2.5-1.5b-q4_k_m | 28 | 1099.4 | 72.5 | 0.907 | 0.893 |

### Time to first token against prompt length (cold prompt cache)

| Model file | Prompt tokens | Time to first token (ms) |
|---|---|---|
| qwen2.5-0.5b-q4_k_m | 40 | 25.3 |
| qwen2.5-0.5b-q4_k_m | 257 | 93.8 |
| qwen2.5-0.5b-q4_k_m | 1469 | 500.4 |
| qwen2.5-0.5b-q8_0 | 40 | 23.8 |
| qwen2.5-0.5b-q8_0 | 257 | 85.5 |
| qwen2.5-0.5b-q8_0 | 1469 | 455.2 |
| qwen2.5-0.5b-fp16 | 40 | 24.6 |
| qwen2.5-0.5b-fp16 | 257 | 83.6 |
| qwen2.5-0.5b-fp16 | 1469 | 434.6 |
| qwen2.5-1.5b-q4_k_m | 40 | 73.1 |
| qwen2.5-1.5b-q4_k_m | 257 | 297.3 |
| qwen2.5-1.5b-q4_k_m | 1469 | 1546.6 |

| Model file | ms per extra token, 40 to 257 | 257 to 1,469 |
|---|---|---|
| qwen2.5-0.5b-q4_k_m | 0.316 | 0.335 |
| qwen2.5-0.5b-q8_0 | 0.284 | 0.305 |
| qwen2.5-0.5b-fp16 | 0.272 | 0.290 |
| qwen2.5-1.5b-q4_k_m | 1.033 | 1.031 |

### Router load imbalance (simulation, popularity assumed, no model run)

| Shape | Popularity | Batch (tokens) | Busiest / mean | Range over trials | Trials |
|---|---|---|---|---|---|
| OLMoE shape (64 experts, 8 per token) | uniform | 8 | 3.67 | 2.00 to 7.00 | 200 |
| OLMoE shape (64 experts, 8 per token) | uniform | 64 | 1.86 | 1.50 to 2.50 | 200 |
| OLMoE shape (64 experts, 8 per token) | uniform | 512 | 1.28 | 1.16 to 1.44 | 200 |
| OLMoE shape (64 experts, 8 per token) | uniform | 4096 | 1.10 | 1.07 to 1.15 | 40 |
| OLMoE shape (64 experts, 8 per token) | zipf | 8 | 7.25 | 5.00 to 8.00 | 200 |
| OLMoE shape (64 experts, 8 per token) | zipf | 64 | 7.11 | 6.12 to 8.00 | 200 |
| OLMoE shape (64 experts, 8 per token) | zipf | 512 | 7.11 | 6.81 to 7.39 | 200 |
| OLMoE shape (64 experts, 8 per token) | zipf | 4096 | 7.11 | 7.00 to 7.17 | 40 |
| Mixtral shape (8 experts, 2 per token) | uniform | 8 | 1.97 | 1.50 to 3.00 | 200 |
| Mixtral shape (8 experts, 2 per token) | uniform | 64 | 1.34 | 1.06 to 1.81 | 200 |
| Mixtral shape (8 experts, 2 per token) | uniform | 512 | 1.11 | 1.04 to 1.24 | 200 |
| Mixtral shape (8 experts, 2 per token) | uniform | 4096 | 1.04 | 1.01 to 1.08 | 40 |
| Mixtral shape (8 experts, 2 per token) | zipf | 8 | 2.69 | 2.00 to 4.00 | 200 |
| Mixtral shape (8 experts, 2 per token) | zipf | 64 | 2.54 | 1.94 to 3.12 | 200 |
| Mixtral shape (8 experts, 2 per token) | zipf | 512 | 2.53 | 2.29 to 2.76 | 200 |
| Mixtral shape (8 experts, 2 per token) | zipf | 4096 | 2.53 | 2.48 to 2.58 | 40 |

### The ten predictions

| # | Prediction | Held | Detail |
|---|---|---|---|
| 1 | Measured decode is below the bound for every model | yes | held |
| 2 | Every model reaches 0.55 to 0.95 of its bound | no | qwen2.5-0.5b-fp16: 0.96 of the bound, outside 0.55 to 0.95 |
| 3 | The 1.5B Q4_K_M reaches a higher fraction of its bound than the 0.5B Q4_K_M | yes | held |
| 4 | Prefill within 1.3x across quantizations while Q4_K_M decodes 1.8x faster than FP16 | yes | held |
| 5 | 1.5B prefill is 0.25 to 0.40 of 0.5B prefill | yes | held |
| 6 | Decode at context 4,096 slows by the extra KV bytes, within 0.07, every model | no | not judged: qwen2.5-0.5b-q4_k_m tg@4096 did not repeat within 10% |
| 7 | Time to first token is linear in prompt length (slopes within 25%) | yes | held |
| 8 | Uniform routing: busiest expert 2.5x the mean at batch 8, at most 1.15x at 4,096 | yes | held |
| 9 | Zipf(1) popularity: busiest expert at least 2x the mean even at batch 4,096 | yes | held |
| 10 | Same bytes per token, same speed: 1.5B Q4_K_M within 25% of 0.5B FP16 | yes | held |
