# Day 6: Prefill, decode and arithmetic intensity

**Stage:** 3, Transformer and MoE inference anatomy. **Label:** M (llama-bench and llama-server on
this machine, four model files) and S (the bytes-per-token bound model, and the mixture-of-experts
router simulation). **Date:** design and predictions 2026-11-26 with the owner present; all timing
2026-11-27 in an unattended scheduled run. **Time spent:** the timed phases ran from 09:38 to 09:53
WAT on 2026-11-27. Hands-on time on 2026-11-26 was not recorded.

## Goal

Predict decode speed from memory bandwidth divided by the bytes read per generated token, and
compare it with measurements across models and quantizations. Show that prefill is compute-bound.
Measure time to first token against prompt length. Simulate how unevenly a mixture-of-experts router
loads its experts.

How the results were judged: ten predictions with fixed thresholds are in the docstring of
`src/inferlab/decode_model.py`. Nine were committed (caa4c74) before any Day 6 timing and before the
three new model files were read; the tenth was added (69f5903) after reading their headers and
before any timing. Not blind: Day 1's figures for the 0.5B Q4_K_M file were already known, and I
estimated predictions 8 and 9 by hand. Eight held, one failed and one could not be judged. Which
files may be cited is in [`results/README.md`](results/README.md).

## Environment

- MacBook Air (Mac15,12): Apple M3, 8 CPU cores, 16 GB, macOS 26.2, Python 3.12.8.
- llama.cpp commit cbb1c1c (`cbb1c1c50bba81929387dea473873d1df87a6212`), Metal, all layers on the
  GPU (`-ngl 99`), f16 KV cache. Every llama-bench entry records `build_commit` cbb1c1c.
- Four model files, all from Qwen's own Hugging Face repositories (`apache-2.0`), checked against
  the published SHA-256 when fetched on 2026-11-26 with the owner's approval (`LOG.md` section 0):

| File | Bytes | sha256 |
|---|---|---|
| `qwen2.5-0.5b-instruct-q4_k_m.gguf` | 491,400,032 | `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` |
| `qwen2.5-0.5b-instruct-q8_0.gguf` | 675,710,816 | `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` |
| `qwen2.5-0.5b-instruct-fp16.gguf` | 1,266,425,696 | `8e0ae26000627ed62de0e78e41860af70094558b9d2913385c842a6aa06cf3fc` |
| `qwen2.5-1.5b-instruct-q4_k_m.gguf` | 1,117,320,736 | `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` |

- Machine state at the start of the bench phase (09:38): AC power, no thermal or performance warning
  recorded, load average 2.22, 2.46 and 2.32, 69% of memory free, no swap; 10 GiB of disk free.
  At the end of the time-to-first-token phase (09:53): load average 1.37, 1.57 and 1.96, 70% free,
  no swap. Docker Desktop was not running. Chrome, Slack and an idle `ollama serve` (0% CPU,
  20 MB) were open.
- Bandwidth for the bound: 87.22 GB/s, Day 3's MLX GPU triad median
  (`days/03-roofline/results/gpu_bandwidth.json`).
- Code: caa4c74 (bound model, router, runner, report, predictions), 69f5903 (headers, router results,
  prediction 10), then today's results, tests and write-up commits.

## What I did

1. On 2026-11-26, with the owner present: wrote the bound model, the router simulation, the runner
   and the report, with tests, and committed the predictions before any timing. Read the four GGUF
   headers (`--phase headers`) and ran the router simulation (`--phase router`).
2. On 2026-11-27 the scheduled run passed pre-flight and ran
   `python scripts/day06_run.py --phase bench`: three rounds, each running every model once as its
   own `llama-bench` process (512-token prefill and 128-token decode, at depth 0 and after 4,096
   tokens, 3 repetitions inside the process). The rounds interleave the models. Each process came
   after a 30 s rest and was bracketed by Day 1's decode probe, with the CPU neighbour sampler
   running. A process's figure is the median of its three repetitions; a cited figure is the median
   of the three processes, and it must repeat with a coefficient of variation of at most 10%.
3. Ran `--phase ttft`: one `llama-server` per model (`-c 4096 --parallel 1`), one warm-up and then
   three requests for each of Day 1's 40, 257 and 1,469-token prompts, with a cold prompt cache
   (no request saw a cached token), bracketed by the same probe.
4. Rendered the tables with `python scripts/day06_report.py`. Nothing was retried and no attempt was
   set aside.

## Results

Every table below is generated from the result files by `scripts/day06_report.py` and embedded
verbatim (a test checks this). "(unstable)" marks a figure whose three runs varied by more than
10%; such a figure is shown but judges nothing.

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

What these numbers show:

- **Bytes per token sets the decode rate, within a factor the bound explains.** All four models
  decode below the bound (prediction 1). The two files that read nearly the same bytes per token,
  the 1.5B Q4_K_M (981.9 MB) and the 0.5B FP16 (989.0 MB), decode at 80.0 and 85.0 tokens/s,
  although one has 1,777 M and the other 630 M parameters (prediction 10). Across the three 0.5B
  quantizations decode falls from 159.8 to 143.8 to 85.0 tokens/s as the bytes read rise from 392.6
  to 525.9 to 989.0 MB.
- **The more bytes a step reads, the closer it gets to the bound.** Measured over bound is 0.72 for
  the 0.5B Q4_K_M, 0.87 for Q8_0, 0.90 for the 1.5B Q4_K_M and 0.96 for FP16. I predicted the 1.5B
  ordering against the 0.5B Q4_K_M (prediction 3, held), with the reason that work done once per
  step weighs less when there is more to read. The ordering fits that reason. I did not measure the
  per-step overhead, so the reason is not shown.
- **Prefill is compute-bound here.** The three 0.5B files prefill 512 tokens at 3170, 3398 and
  3590 tokens/s, within the 1.3x I allowed, while their decode rates differ by much more
  (prediction 4). The 1.5B file, with more parameters to multiply, prefills at 985 tokens/s, inside
  the 0.25 to 0.40 of the 0.5B Q4_K_M's rate I predicted (prediction 5). Prefill runs 12.3 to 42.2
  times faster than decode per token, which fits a 512-token batch sharing each weight read across
  512 tokens; the reuse itself was not measured.
- **The lowest-bit file prefills slowest.** Among the 0.5B files the order is FP16 (3590), Q8_0
  (3398), Q4_K_M (3170): the reverse of decode. I did not predict the order. A likely cause is the
  cost of unpacking quantized blocks during the matrix multiply, but I did not measure that.
- **Deep context slows decode by about what the extra KV bytes predict.** After 4,096 tokens the
  measured deep-over-start ratios are 0.898 to 0.957 against predicted 0.886 to 0.952, each within
  0.025 of its prediction. Prediction 6 still counts as not judged, because the 0.5B Q4_K_M's deep
  figure did not repeat within 10% (next section).
- **Time to first token is linear in prompt length.** For every model the cost per extra prompt
  token between 257 and 1,469 tokens is within 25% of that between 40 and 257 (prediction 7): for
  example 0.316 and 0.335 ms per token for the 0.5B Q4_K_M and 1.033 and 1.031 for the 1.5B. The
  0.5B Q4_K_M's figures (25.3, 93.8 and 500.4 ms) are close to Day 1's (26.0, 92.9 and 490.8 ms),
  on the same build and file.
- **Router imbalance (simulation, label S).** With uniform routing over 64 experts and 8 per token,
  the busiest expert carries 3.67 times the mean load at a batch of 8 tokens and 1.10 times at 4,096
  (prediction 8). With Zipf(1) popularity it stays at 7.11 times the mean at 4,096 (prediction 9):
  batching smooths randomness, not skew. These are a model of routing with assumed popularity. No
  mixture-of-experts model ran.

## What failed or surprised me

- **Prediction 2 failed, narrowly.** I predicted every model would reach 0.55 to 0.95 of its bound.
  The 0.5B FP16 reached 0.964 (the table rounds it to 0.96). So either Day 3's triad figure
  understates the bandwidth a pure read stream gets, or the model overstates the bytes read. I did
  not measure which. The threshold was not changed.
- **Prediction 6 could not be judged.** The 0.5B Q4_K_M's decode after 4,096 tokens was 145.3 and
  149.1 tokens/s in rounds 0 and 1 and 103.8 in round 2, a coefficient of variation of 15.5%. By the
  rule fixed before the run, an unstable figure judges nothing, so the report marks prediction 6
  "not judged", which counts as not held. The other three models agree with the prediction, as above.
  I did not rerun the measurement, because choosing to repeat after seeing a result would bias it.
- **The machine slowed during round 2.** The decode probe after every round 2 process read 144.7 to
  152.7 tokens/s, against 164.1 to 166.1 before them; in rounds 0 and 1 the after-probes were 162.7
  to 166.5 (`bench.json`, `gates`). All probes were within the 20% gate,
  so nothing was refused. The neighbour sampler did not flag contention in round 2. A fanless
  machine warming up after ten minutes of GPU work would fit (Day 1 saw this), but I did not record
  temperatures, so this is a guess.
- **One process ran with a busy neighbour.** During round 0 of the 0.5B Q4_K_M a process listed as
  `deleted` peaked at 96.7% CPU and the sampler marked the run contended. That run's figures are
  inside the spread of the other two rounds and were kept, as the protocol says.
- **Prefill order is the reverse of decode order** across quantizations, as above.

## Not claimed

- **Why any model falls short of its bound**, or why FP16 comes so close. Per-step overhead and the
  read bandwidth a decode step actually achieves were not measured.
- **That quantized prefill is slower because of unpacking.** Only the order was measured.
- **Anything about a real mixture-of-experts model.** None was run: the smallest one considered
  (2.56 GB) would have broken the runbook's disk rule on 2026-11-26. The router rows are a
  simulation with assumed popularity.
- **Decode at long context for the 0.5B Q4_K_M.** The figure is shown as unstable and cited nowhere
  else.
- **Batch sizes above one, other engines, other builds, other chips.**

## Limits

- One machine, one llama.cpp build, one model family, one day, f16 KV cache, flash attention as
  llama-bench sets it by default.
- The bound uses Day 3's triad bandwidth, measured with MLX, a different program from llama.cpp.
- Time to first token was measured with three requests inside one server process per model, not
  three processes. Its repeats agree closely (every coefficient of variation is at most 1.2%), but
  they share a process, so they are less independent than the bench runs.
- The prompts are Day 1's, synthetic and public.
- The predictions were fixed before the timing; one failed and one could not be judged, and none was
  changed. Prediction 10 was added after the headers were read and before any timing.

## Evidence ledger entries

The rows added to [EVIDENCE.md](../../EVIDENCE.md) today:

- Decode against the bytes-per-token bound for four files, and the two files with the same bytes
  reaching nearly the same speed (M, S).
- Prefill across quantizations and sizes, compute-bound, and its reverse order (M).
- Decode after 4,096 tokens of context against the KV prediction, with the unstable figure named (M, S).
- Time to first token, linear in prompt length, for four files (M).
- Router load imbalance, uniform and Zipf (S).
- The ten predictions: eight held, one failed, one not judged.
- Built things: the bound model, the router, the runner, the report, their tests and the
  deliberate breakages added today.
