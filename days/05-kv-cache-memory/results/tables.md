### Weights: the calculator against the model file

| Quantity | Parameters |
|---|---|
| Calculator, embeddings tied as the config says | 494,032,768 |
| Calculator, output matrix counted separately (added after prediction 1 failed) | 630,167,424 |
| Elements in the GGUF file's 291 tensors | 630,167,424 |
| llama.cpp's printed `model params` | 630.17 M |
| Vocabulary x hidden size (one embedding matrix) | 136,134,656 |

| Type | Tensors | Elements | Bytes | Bits per weight |
|---|---|---|---|---|
| Q5_0 | 133 | 387,989,504 | 266,742,784 | 5.50 |
| Q8_0 | 13 | 137,510,912 | 146,105,344 | 8.50 |
| Q6_K | 12 | 52,297,728 | 42,900,480 | 6.56 |
| Q4_K | 12 | 52,297,728 | 29,417,472 | 4.50 |
| F32 | 121 | 71,552 | 286,208 | 32.00 |
| All | 291 | 630,167,424 | 485,452,288 | 6.16 |

Offset check (added after prediction 3 held): 290 of the 290 gaps between consecutive tensor offsets equal the block table's size for the earlier tensor, padded to the alignment, and the last tensor ends at the end of the file.

| Weights | Bytes | MiB |
|---|---|---|
| Config's 494,032,768 parameters at 4.5 bits (Q4_K's own rate) | 277,893,432 | 265.0 |
| Tensor data in the file | 485,452,288 | 463.0 |
| Of which the two embedding matrices (`token_embd.weight`, `output.weight`) | 238,235,648 | 227.2 |

The file holds 1.75 times the naive estimate, and the two embedding matrices are 49% of it.

### KV cache: the calculator against what llama.cpp reports

| Context asked | Cache type | Cells reported | llama.cpp KV (MiB) | Calculator at those cells (MiB) | Difference | Calculator at the context asked (MiB) | Launches |
|---|---|---|---|---|---|---|---|
| 512 | f16 | 512 | 6.00 | 6.00 | +0.00% | 6.00 | 3 |
| 1000 | f16 | 1024 | 12.00 | 12.00 | +0.00% | 11.72 | 3 |
| 4096 | f16 | 4096 | 48.00 | 48.00 | +0.00% | 48.00 | 3 |
| 32768 | f16 | 32768 | 384.00 | 384.00 | +0.00% | 384.00 | 3 |
| 512 | q8_0 | 512 | 3.19 | 3.19 | +0.08% | 3.19 | 3 |
| 1000 | q8_0 | 1024 | 6.38 | 6.38 | +0.08% | 6.23 | 3 |
| 4096 | q8_0 | 4096 | 25.50 | 25.50 | +0.00% | 25.50 | 3 |
| 32768 | q8_0 | 32768 | 204.00 | 204.00 | +0.00% | 204.00 | 3 |
| 512 | q4_0 | 512 | 1.69 | 1.69 | +0.15% | 1.69 | 3 |
| 1000 | q4_0 | 1024 | 3.38 | 3.38 | +0.15% | 3.30 | 3 |
| 4096 | q4_0 | 4096 | 13.50 | 13.50 | +0.00% | 13.50 | 3 |
| 32768 | q4_0 | 32768 | 108.00 | 108.00 | +0.00% | 108.00 | 3 |

### Memory breakdown llama.cpp prints at shutdown

| Context asked | Cache type | Model (MiB) | Context (MiB) | Compute (MiB) | KV line (MiB) |
|---|---|---|---|---|---|
| 512 | f16 | 462 | 6 | 43 | 6.00 |
| 1000 | f16 | 462 | 12 | 44 | 12.00 |
| 4096 | f16 | 462 | 48 | 50 | 48.00 |
| 32768 | f16 | 462 | 384 | 106 | 384.00 |
| 512 | q8_0 | 462 | 3 | 43 | 3.19 |
| 1000 | q8_0 | 462 | 7 | 44 | 6.38 |
| 4096 | q8_0 | 462 | 26 | 50 | 25.50 |
| 32768 | q8_0 | 462 | 204 | 106 | 204.00 |
| 512 | q4_0 | 462 | 2 | 43 | 1.69 |
| 1000 | q4_0 | 462 | 4 | 44 | 3.38 |
| 4096 | q4_0 | 462 | 14 | 50 | 13.50 |
| 32768 | q4_0 | 462 | 108 | 106 | 108.00 |

### Parameters: the calculator against the published figures

| Model | Total, calculator (B) | Total, published (B) | Within rounding | Active, calculator (B) | Active, published (B) | Within rounding  | Non-embedding, calculator (B) | Non-embedding, published (B) |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.4940 | 0.49 | yes | n/a | n/a | n/a | 0.3579 | 0.36 |
| Qwen2.5-7B-Instruct | 7.6156 | 7.61 | no | n/a | n/a | n/a | 6.5256 | 6.53 |
| Mixtral-8x7B-v0.1 | 46.7028 | 46.7 | yes | 12.8799 | 12.9 | yes | 46.4406 | n/a |
| OLMoE-1B-7B-0924 | 6.9192 | 6.9 | yes | 1.2820 | 1.3 | yes | 6.7131 | n/a |
| DeepSeek-V2-Lite | 15.7065 | 15.7 | yes | 2.6612 | 2.4 | no | 15.2871 | n/a |

### What the calculator says for the five models (a calculation, not a measurement)

| Model | Weights at 16 bits (GiB) | KV per token, f16 (KiB) | KV at 4,096 tokens (GiB) | KV at 32,768 tokens (GiB) | KV at 131,072 tokens (GiB) |
|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.92 | 12.0 | 0.05 | 0.38 | 1.50 |
| Qwen2.5-7B-Instruct | 14.19 | 56.0 | 0.22 | 1.75 | 7.00 |
| Mixtral-8x7B-v0.1 | 86.99 | 128.0 | 0.50 | 4.00 | 16.00 |
| OLMoE-1B-7B-0924 | 12.89 | 128.0 | 0.50 | 4.00 | 16.00 |
| DeepSeek-V2-Lite (compressed cache) | 29.26 | 30.4 | 0.12 | 0.95 | 3.80 |
| DeepSeek-V2-Lite (expanded cache) | 29.26 | 270.0 | 1.05 | 8.44 | 33.75 |

### The ten predictions

| # | Prediction | Held | Detail |
|---|---|---|---|
| 1 | Parameter count equals the GGUF's tensor elements exactly and llama.cpp's printed one | no | calculator 494032768 parameters, the file's tensors hold 630167424 (-136134656) |
| 2 | The calculator's tensors are exactly the GGUF's tensors, names and shapes | no | in the file but not expected: output.weight (896, 151936) |
| 3 | Block-table tensor sizes add to the file's tensor-data region within 0.01% | yes | held |
| 4 | KV size within 0.5% of llama.cpp at 512, 4096 and 32768 for f16, q8_0, q4_0 | yes | held |
| 5 | 1000 tokens of context are reported as 1024 cells, 2% or more above the 1000 figure | yes | held |
| 6 | The reported KV size is identical across the three launches of each configuration | yes | held |
| 7 | Breakdown context column within 2 MiB of KV; model columns within 2% of tensor data | yes | held |
| 8 | Total parameters of all five models within the rounding of the published figure | no | Qwen2.5-7B-Instruct: calculator 7.6156 B total, published 7.61 B (+0.0056) |
| 9 | Activated parameters of the three mixture-of-experts models within published rounding | no | DeepSeek-V2-Lite: calculator 2.6612 B active, published 2.4 B (+0.2612) |
| 10 | DeepSeek-V2-Lite's compressed cache is 8 to 10 times smaller than the expanded one | yes | held |
