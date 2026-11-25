# Day 5: KV cache and memory calculator

**Stage:** 3, Transformer and MoE inference anatomy. **Label:** M (llama.cpp's own memory reports and
the model file's header, on this machine), S (the calculator, on five model configurations) and L
(the published parameter counts it is compared with). **Date:** 2026-11-25. **Time spent:** elapsed
from about 11:20 WAT with the owner present, after the scheduled 09:22 run stopped at its disk
check. Hands-on time was not recorded.

## Goal

Build a calculator that sizes the memory a language model needs, weights plus KV cache, from its
Hugging Face `config.json`. Check it against five real configurations (grouped-query attention, two
mixtures of experts, and multi-head latent attention with experts), against the parameter counts the
models' own cards and papers print, and against what llama.cpp reports for the one model that runs
on this machine. Say where it is right and where it is not.

How the results were judged: ten predictions were written in `src/inferlab/kvmem.py` and committed
(4a95943) before any llama.cpp run, before my GGUF reader touched the real model file and before the
published figures were looked up. Predictions 8 to 10 are not blind, because I worked those parameter
counts out by hand while designing the module. Four of the ten failed. Which files may be cited is
in [`results/README.md`](results/README.md).

## Environment

- MacBook Air (Mac15,12): Apple M3, 8 CPU cores, 16 GB, macOS 26.2, Python 3.12.8.
- llama.cpp commit cbb1c1c (`cbb1c1c50bba81929387dea473873d1df87a6212`), Metal. Model
  `qwen2.5-0.5b-instruct-q4_k_m.gguf`, 491,400,032 bytes, sha256
  `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`. Every server ran with
  `-c N --parallel 1 -ngl 99 -ctk T -ctv T -fa on`, so flash attention was on for all of them.
- The matrix ran from 11:46 to 11:48 WAT. AC power; no thermal or performance warning recorded;
  48% of memory free; 1.7 GB of swap in use; 9.46 GB of disk free; GPU use at the start check peaked
  at 9%. Load average was 3.49, 12.55 and 11.62 (1, 5 and 15 minutes). The 5 and 15 minute figures
  were high and I did not find the cause. Nothing is timed today, so this matters less than on
  earlier days: the figures are sizes llama.cpp says it allocated.
- Disk was the obstacle. The scheduled run at 09:22 stopped at 7.3 GiB free. I quit Chrome normally
  and cleared the Go build cache (`go clean -cache`, 4.63 GiB, regenerable), which gave 12.95 GiB.
  Within the hour free disk fell below 8 GB again (`~/Library/Caches` had grown from about 2.9 to
  6.6 GiB and a Go linker was running; I did not confirm what refilled it), and the runner's
  start-of-run guard refused its first attempt, so that attempt produced no results.
- No decode probe was taken, because nothing is timed (DAILY.md's probe rule is for timed GPU
  phases).
- Code: 4a95943 (calculator, reader, predictions), cf9692d (log parser, runner, published figures),
  f4c4262 (parser fix, per-prediction checks, report), 260edc8 (results, analysis, tests, mutations),
  45c40ac (the per-tensor offset check), and the write-up commit after it.
- The five configuration files are pinned by commit and sha256 in `configs/manifest.json`
  (b97800c). The published figures in `published.json` were read from the pages named there on
  2026-11-25 through a web-fetch agent; I did not open the pages myself.

## What I did

1. Wrote the calculator `kvmem.py`, the GGUF header reader `gguf.py` and the prediction checks
   `kvcheck.py`, with tests, and committed them with the ten predictions before anything was
   measured. The calculator supports four architectures and refuses everything else, including
   `llama`, `mistral` and any sliding-window configuration, because none of those was checked.
2. Pinned each model's parameter count by hand in the tests. The hand count for DeepSeek-V2-Lite was
   800,000 too high (I mis-added 26 layers) and the test caught it. I confirmed the calculator's
   figure by listing all 5,291 of that model's tensors one by one in a separate script.
3. Read the published figures, then judged predictions 8 and 9 before llama.cpp ran. Both failed.
4. Before the matrix I also launched the server once, to learn what its JSON log looks like, and ran
   my reader on the real file. The single launch showed a 6.00 MiB cache at 512 tokens and
   `model params = 630.17 M`, so the first cell of prediction 4 is not blind. The reader showed that
   predictions 1 and 2 had failed and prediction 3 had held. Details are in `LOG.md` section 3.
5. Wrote the parser and the runner. The runner starts each server itself, waits for `/health`, stops
   it with SIGINT so that it prints its memory breakdown, and parses the log. Attempt 1 was refused
   at the start (disk under 8 GB). Attempt 2 ran all 36 launches and my parser marked all 36 unusable
   because it looked for a flash attention line that only `-fa auto` prints; the attempt is in
   `results/rejected/`. Attempt 3, from committed code, is the one cited: 12 configurations (four
   contexts, three cache types), three launches each, none unusable.
6. Rendered the tables from the result files (`python scripts/day05_gguf.py`, then
   `python scripts/day05_report.py`), wrote tests that recompute the analysis from the committed
   inputs, and added 46 deliberate breakages of the new code to the mutation script.

## Results

Each KV figure is the same in all three launches of its configuration. The model file was read three
times with identical results. Sizes are what llama.cpp reports, in MiB, not what the operating system
shows for the process. Section "What these numbers show" says what they do and do not support.

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

Using the calculator on a model that is not on this machine (a calculation, not a measurement):

```
python -m inferlab.kvmem days/05-kv-cache-memory/configs/Qwen2.5-7B-Instruct.config.json \
    --context 32768 --sequences 4 --bits-per-weight 4.5 --budget-gib 16
```

What these numbers show:

- **The KV cache formula matches llama.cpp at all 12 settings.** At every context that is a multiple
  of 256 the calculator and the KV line agree to the printed precision. The largest difference is
  +0.15% (q4_0 at 512 tokens: llama.cpp prints 1.69 MiB, the calculator gives 1.6875, which prints
  as 1.69). At 32,768 tokens the q8_0 and q4_0 caches are 204.00 and 108.00 MiB against 384.00 for
  f16, exactly 17/32 and 9/32 of it.
- **llama.cpp rounds the context up.** The one context that is not a multiple of 256 that I tried,
  1000, was reported as 1,024 cells at all three cache types, which makes the cache 2.4% larger than the
  calculator's figure for 1000 tokens (for f16, 12.00 MiB against 11.72). I did not test other contexts, so the
  rule "round up to a multiple of 256" is a guess from this one case and the multiples tested.
- **The calculator was wrong about the file, not about the model.** The config ties the input and
  output embeddings, so the model has 494,032,768 parameters. The GGUF file stores a separate
  `output.weight` (896 x 151,936, Q8_0) next to the embedding (Q5_0), so its 291 tensors hold
  630,167,424 elements, and llama.cpp prints exactly that as `model params`. The difference is one
  embedding matrix, 136,134,656 elements. Counting the output matrix separately reproduces the file
  exactly. Whether other GGUF files do the same I did not check.
- **That matrix is why a naive weight estimate is far off for a small model.** The config's
  parameters at Q4_K's own rate of 4.5 bits would be 265.0 MiB. The file's tensor data is 463.0 MiB,
  1.75 times as much, at 6.16 bits per weight overall, and the two embedding matrices are 49% of it.
- **A file named Q4_K_M is a mixture here.** It holds weights in Q5_0 (133 tensors), Q8_0, Q6_K and
  Q4_K, and float32 for 121 small tensors. In this file a matrix is a K-quant type exactly when its
  row length is a multiple of 256: 24 of the 170 weight matrices, which are the 12 Q4_K and 12 Q6_K
  tensors. That is an observation about this file. I did not read llama.cpp's quantization rules.
- **The block-size table is right for the five types present.** Adding every tensor's size from it,
  with each padded to the file's 32-byte alignment, gives 485,452,288 bytes, the exact size of the
  file's tensor-data region. I then added a stricter check, because a sum could hide two errors that
  cancel: each of the 290 gaps between consecutive tensor offsets equals the table's size for the
  earlier tensor, for all five types, and the last tensor ends at the end of the file.
- **llama.cpp's own breakdown matches the file and the cache.** Its model column is 462 MiB at every
  setting (the data region is 462.96 MiB), and its context column is within 1 MiB of the KV line (the widest gap is 7 against 6.38, for a
  q8_0 cache at 1,000 tokens).
  The third column, compute buffers, is not in the calculator: it is 43 MiB at 512 tokens and
  106 MiB at 32,768, the same for all three cache types at each context. At 32,768 tokens with an f16
  cache it is 28% of the KV cache. The calculator's total therefore undercounts what llama.cpp
  allocates, by an amount that grows with context.
- **Parameter counts match the published figures where the published figure is precise enough.**
  Four of five totals are within the rounding of what the card or paper prints, as are the activated
  parameters of Mixtral and OLMoE. The two that are not are in the next section.
- **The cache follows attention, not the number of experts.** Mixtral (86.99 GiB of weights at 16
  bits) and OLMoE (12.89 GiB) both need 128.0 KiB per token. Layers times KV heads times head size
  is the same for the two (32 x 8 x 128 and 16 x 16 x 128), and the experts change the weights only.
  This is a calculation: no mixture of experts model ran here.
- **Latent attention shrinks the cache about 8.9 times, by calculation.** DeepSeek-V2-Lite needs
  30.4 KiB per token with the compressed cache and 270.0 KiB if the per-head keys and values are
  stored. Which one an engine stores is engine specific, and no engine was run for this model.

## What failed or surprised me

- **Predictions 1 and 2 failed because I assumed the file mirrors the model.** I predicted no
  separate output matrix, because the model ties its embeddings. The file has one, so the calculator's
  count is 136,134,656 short of the file's, and the only tensor that differs is `output.weight`; the
  other 290 match by name and shape. The two embedding tensors have different types (Q5_0 and Q8_0),
  so they are not byte copies. I did not compare their values.
- **Prediction 8 failed narrowly on one model.** For Qwen2.5-7B the calculator gives 7.6156 B,
  which rounds to 7.62, and the card prints 7.61. My rule was half a unit of the last printed digit
  (0.005) and the difference is 0.0056. I did not find out why the card prints 7.61, and I did not
  change the rule. The same card's non-embedding figure (6.53) is within rounding of the calculator's
  6.5256, which I did not predict.
- **Prediction 9 failed by a wide margin on one model.** DeepSeek-V2-Lite's calculated activated
  parameters are 2.6612 B against a published 2.4 B. The calculator counts the embedding tables as
  active and I did not find which convention the authors use, so I cannot say whether the calculator
  or the convention is the cause. Mixtral and OLMoE are within rounding.
- **A weight estimate from the quantization name is not safe.** I expected a Q4_K_M file to cost
  roughly the Q4_K rate per weight. It cost 1.75 times that, for the reasons above. The calculator
  takes bits per weight as an input and does not know a file's real mixture.
- **My own parser rejected a good run.** I wrote the "flash attention is on" rule from the one log I
  had, an auto-mode log. With `-fa on` the line differs, so all 36 launches of attempt 2 were marked
  unusable. The measurement was fine: attempt 2 and the cited run agree on every figure (a test pins
  this). I fixed the parser with a test and reran everything from committed code.
- **The disk guard fired and it should have.** Free disk dropped under 8 GB between my last check
  and the first run. The runner refused, no partial results existed, and I reran once the space came
  back.

## Not claimed

- **Anything about speed.** Nothing was timed.
- **That the calculator's total predicts the process's resident memory.** I did not measure the
  operating system's view of the process, only what llama.cpp reports. The compute buffers the
  calculator leaves out are part of the gap.
- **That other GGUF files store a separate output matrix**, or that llama.cpp rounds every context up
  to a multiple of 256.
- **Why Qwen2.5-7B's card prints 7.61 B, or which convention gives 2.4 B activated.**
- **Anything about an engine's memory use for a mixture of experts or latent attention model.** Only
  the 0.5B grouped-query model ran. Those rows are calculations checked against configurations and
  published parameter counts, and nothing else.
- **Contexts beyond a model's trained length.** The 131,072 column in the scenario table is
  arithmetic. OLMoE was trained for 4,096 tokens and Qwen2.5-7B needs a scaling method past 32,768.
- **Cache types other than f16, q8_0 and q4_0.** `bf16` and `f32` are in the calculator's table and
  were not compared with llama.cpp.

## Limits

- One model, one llama.cpp build, one machine, one day. The machine was not quiet (the load averages
  above), though nothing here is a timing.
- The calculator supports four architectures. It refuses `llama`, `mistral` and sliding-window
  attention, so it cannot size many popular models without more work and more validation.
- Flash attention was on and `--parallel 1` throughout. A different slot count or an unfused
  attention path might report different buffers; the KV size is not expected to change, but I did not
  test it.
- The published figures come from model cards and papers, read through a web-fetch agent, some pages
  partly. The figures used were quoted from the pages as printed. Where a page printed a coarser
  figure as well (OLMoE's card says 1B active and 7B total, DeepSeek-V2-Lite's card table says 16B),
  the more precise figure was used and the coarser one is kept in `published.json`.
- The prediction thresholds were fixed before the results. Four predictions failed, and none was
  changed afterwards. Two columns and one option were added after seeing a failure and are labelled
  as such: the calculator row with the output matrix counted separately, and the
  `--output-matrix-separate` option of the command line.

## Evidence ledger entries

The rows added to [EVIDENCE.md](../../EVIDENCE.md) today:

- The KV cache formula against llama.cpp's reports at 12 settings, and the rounding up of the
  context (M).
- The model file's real contents against the calculator: a separate output matrix, 6.16 bits per
  weight, embeddings 49% of the file, and the block-size table (M).
- llama.cpp's memory breakdown, including compute buffers that grow with context (M).
- The calculator against published parameter counts, with the two failures disclosed (S, L).
- Per-token cache sizes for grouped-query, mixture-of-experts and latent attention models, as
  calculations (S).
- The ten predictions, six held and four did not (M, S, L).
- Built things: the calculator, the GGUF reader, the checks, the log parser, the runner, the report
  and their tests, and 46 more deliberate breakages.
