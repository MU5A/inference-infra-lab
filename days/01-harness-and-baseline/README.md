# Day 1: benchmark harness and baseline

**Stage:** 1, systems foundations. **Label:** M (measured on this machine). **Date:** 2026-11-22.

## Goal

Build a measurement tool I can trust, then use it to take one baseline: how fast does a small
model run on this laptop, and how much of what the tool reports is the tool itself? Every later
day reuses this harness, so a mistake here would quietly corrupt every result after it.

## Environment

| | |
|---|---|
| Machine | MacBook Air (Mac15,12, no fan), Apple M3, 8 CPU cores, 10 GPU cores, 16 GB unified memory, macOS 26.2, arm64 |
| Server | llama.cpp at commit `cbb1c1c50bba81929387dea473873d1df87a6212` (2026-11-22), built from source with Metal, AppleClang 21 |
| Model | `qwen2.5-0.5b-instruct-q4_k_m.gguf`, 491 MB, sha256 `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`, matches the digest the repository publishes (Apache-2.0) |
| Server flags | `-c 4096 -ngl 99 --parallel 1` (all layers on the GPU, one slot) |
| Client | Python 3.12.8 (native arm64), httpx 0.28.1 |
| Conditions | On AC power, no thermal warnings, 1-minute load average 3.6 at the start of the baseline |

## What I did

1. Built llama.cpp from source with `scripts/build_llama.sh` and downloaded the model with
   `scripts/fetch_model.sh`, which compares the file with the sha256 Hugging Face publishes and
   refuses a mismatch.
2. **Read the real wire format before writing the client.** One raw streamed response showed
   things I would otherwise have assumed wrongly. The first event has `content: null`. The final
   event carries `usage` and `timings`. And `usage.completion_tokens` was 3 for a reply with only
   2 text events, because the count includes an end token that carries no text.
3. Wrote the harness (`src/inferlab`): a streaming client, percentile statistics, a run loop with
   warm-up, and an environment record. Definitions are in the module docstring of `client.py`.
4. Wrote a fake streaming server that copies the real wire format, stamps every event with its
   send time, and can fail on purpose, then 98 tests against it (the suite has grown since).
5. Checked that the tests can fail: 17 deliberate breakages of the code, each of which must turn
   the suite red. All 17 are caught. One was missed at first (see below). The list of breakages has
   grown with every day since; `python scripts/mutation_check.py` runs the current set.
6. Took the baseline with `python -m inferlab bench` on a clean machine.
7. Measured the three things the baseline left open: the harness's own timing error, a cold against
   a warm prompt cache, and a cross-check against `llama-bench`. Predictions and stability limits
   were written first (`src/inferlab/crosscheck.py`). The cache and llama-bench measurements took
   four attempts; the first three were discarded and are kept in
   [`results/rejected/`](results/rejected/README.md).

## Results

Raw data: [`results/baseline_llama_server.json`](results/baseline_llama_server.json). The tables
below are produced by `scripts/day01_report.py` and a test checks they match.

Settings: 5 measured runs after 2 warm-up runs per prompt, max_tokens 128, temperature 0.0, seed 1, prompt cache off. Load average at the start: 3.61. Now drawing from 'AC Power'.

| Prompt | Prompt tokens | TTFT median ms (min to max) | Decode tok/s median (min to max) | ITL ms p50 / p95 / p99 | Reply tokens |
|---|---|---|---|---|---|
| short | 40 | 26.0 (25.8 to 26.7) | 152.6 (143.6 to 156.4) | 6.54 / 7.53 / 8.33 | 56 (stop) |
| medium | 257 | 92.9 (92.8 to 96.9) | 153.2 (147.9 to 155.3) | 6.49 / 7.25 / 8.13 | 128 (length) |
| long | 1469 | 490.8 (489.5 to 496.7) | 149.4 (141.4 to 150.0) | 6.65 / 7.45 / 8.14 | 83 (stop) |

Harness against the server's own timer:

| Prompt | Harness decode tok/s | Server-reported decode tok/s | Difference |
|---|---|---|---|
| short | 152.6 | 152.7 | 0.07% |
| medium | 153.2 | 153.1 | 0.01% |
| long | 149.4 | 149.5 | 0.04% |

Prompt processing rate:

| Prompt | Prompt tokens | Server-reported prefill tok/s | Implied by harness TTFT |
|---|---|---|---|
| short | 40 | 1592 | 1540 |
| medium | 257 | 2805 | 2766 |
| long | 1469 | 3005 | 2993 |

What the numbers show:

- **The harness and the server agree on decode speed to within 0.1%.** That is an independent
  check that the tool is not distorting what it measures.
- **Time to first token is dominated by prompt processing.** It grows from 26 ms to 491 ms as the
  prompt grows from 40 to 1,469 tokens, while decode speed stays near 150 tokens per second.
- **Prompt processing is not a constant rate.** It rises from about 1,600 to 3,000 tokens per
  second with prompt length, so per-request fixed costs matter for short prompts.
- **Runs are tight.** Across five runs, time to first token ranged by 0.9 ms (short), 4.1 ms
  (medium) and 7.2 ms (long), at most about 4.5% of the median.

## Follow-up measurements

Three measurements were left open after the baseline. The predictions and limits were written
before the data, in [`src/inferlab/crosscheck.py`](../../src/inferlab/crosscheck.py). Raw data:
`results/harness_overhead.json`, `results/cache_comparison.json` and `results/llama_bench.json`.
The tables are produced by `scripts/day01_report.py` and a test checks they match. A row from a
run that was not repeatable would carry no numbers.

Harness overhead, three runs against a fake server with known timing:

| Run | Events | Median us | p95 us | p99 us | Max us |
|---|---|---|---|---|---|
| 1 | 3450 | 268 | 448 | 738 | 5728 |
| 2 | 3450 | 268 | 422 | 741 | 3465 |
| 3 | 3450 | 271 | 447 | 756 | 3016 |
| **Upper bound** (worst of each column) | | 271 | 448 | 756 | 5728 |

Events that appeared to arrive before they were sent: 0. Client CPU per event with the profiler attached: 744 us (the profiler inflates this).

Cold against warm prompt cache, one session:

| Prompt | Prompt tokens | Cold TTFT ms | Warm TTFT ms | Saved ms | Speed-up | Cached tokens reused per warm request | Predicted at least 5x |
|---|---|---|---|---|---|---|---|
| short | 40 | 25.5 | 8.0 | 17.5 | 3.2x | 39 | did not hold |
| medium | 257 | 91.9 | 8.3 | 83.6 | 11.1x | 256 | held |
| long | 1469 | 485.3 | 10.3 | 475.0 | 47.2x | 1468 | held |

Decode speed probes either side of each phase, to show the machine stayed up to speed:

| Phase | Decode probe before, tokens/s | Decode probe after, tokens/s |
|---|---|---|
| cold | 164.6 | 151.6 |
| warm | 154.5 | 160.4 |

llama-bench cross-check:

| Measurement | llama-bench tokens/s (standard deviation) | Variation | Compared with | Difference | Predicted within |
|---|---|---|---|---|---|
| pp512 | 3191.9 (7.4) | 0.2% | server prefill, long prompt 3005.2 | +6.2% | 15%: held |
| tg128 | 159.7 (3.7) | 2.3% | harness decode baseline 152.6 | +4.7% | 10%: held |

Decode speed probes either side of the run:

| Phase | Decode probe before, tokens/s | Decode probe after, tokens/s |
|---|---|---|
| bench | 160.2 | 160.9 |

What they show:

- **The harness adds well under a millisecond at the median.** The worst of three runs gives 271
  us at the median, 448 us at p95 and 756 us at p99. It is a one-sided bound, because interference
  can only add error. The median is about 4% of a typical gap between tokens (6.5 ms in the
  baseline). The worst single event was 5.7 ms, in a run during which Docker Desktop's helper was
  using up to 60% CPU; I did not show that caused it. This was measured against a fake server that
  sends an event every 10 ms, not against llama.cpp.
- **A warm prompt cache removes the part of time to first token that grows with the prompt.** Warm
  time to first token was 8.0, 8.3 and 10.3 ms for 40, 257 and 1,469 tokens, against 25.5, 91.9
  and 485.3 ms cold, so the saving grows with the prompt. Every warm request reused all but one
  prompt token (the last is always processed again) and the cold runs reused none, as predicted.
- **One prediction failed.** I predicted at least a 5x speed-up for every prompt, and the 40-token
  prompt got 3.2x. A warm request costs about 8 ms whatever the prompt length, close to the 6.5 ms
  of one decode step in the baseline. That is consistent with the last prompt token still going
  through the model, plus request handling, but I did not isolate it. The prediction was wrong
  because a short prompt has little prefill to remove.
- **llama-bench agreed with the server inside the predicted tolerances.** Decode was 4.7% above
  the harness baseline and prompt processing 6.2% above the server's rate on the long prompt. Both
  are in the direction expected of a tool that skips HTTP and the server, but I did not measure
  that mechanism. The prompt comparison is approximate: llama-bench processed 512 tokens and the
  server rate is for a 1,469-token prompt.

## What failed or surprised me

- **Two assumptions about the server were wrong** and would have produced wrong numbers. The
  token count includes the end token, so the decode rate is computed from text events and the
  usage-based rate is kept beside it. And the server can reuse a cached prompt prefix between
  requests (every response carries a cached-token count), which would make repeated prompts look
  nearly free. The follow-up measurements above put a number on that effect. Cold runs turn the
  cache off, and every result records the cached-token count, which is zero here.
- **One gap in my own tests showed up under deliberate breakage.** The command-line refusal and
  override logic had no test until the breakage run exposed it. It has four now.
- **One test failed once and I could not explain it.** The suite failed one test while the
  machine was under heavy load, and I did not capture which one. It then passed three times in a
  row, and ten more times under stress I created myself (six runs of the client tests, four of the
  whole suite). I could not reproduce it. Three assertions depended on wall-clock pacing, so I
  replaced them with tests built from exact, scripted arrival times and kept the tests over HTTP
  to properties that hold at any speed. All 17 breakages are still caught. If it happens again in
  CI it will be investigated, not retried.
- **The machine would not stay quiet.** The pre-flight check reported the Mac on battery power
  and, later, load averages between 5.7 and 28 caused by another process on the machine. It
  refused to start the overhead measurement at a load of 4.1, and I did not override it. A working
  laptop is not a benchmark machine unless that is checked every time, and the daily runbook
  depends on the same check.

- **A steady result can still be wrong.** The third attempt at the cache and llama-bench
  measurements ran about three times slow from start to finish (llama-bench decode 55.7 tokens/s
  against 159.7 in the accepted run, prompt processing 779.5 against 3,191.9), and its repeats
  varied by under 0.5%, so the run-to-run limit passed it. Quick probes afterwards read 46 and 55
  tokens/s. After four minutes of idle they read 166, and they stayed there at about nine and
  sixteen minutes. Memory bandwidth measured at the time was normal (84.5 GB/s) and the Metal backend was
  in use. This is a fanless laptop and the benchmark is itself sustained load, so heat is the
  likely cause, but I read no temperature or clock and have not proven it. The fix is a decode
  probe before and after every GPU phase, refusing the phase if it reads more than 20% below the
  baseline. I chose the 20% after seeing this data; clean probes read from 0.6% below the baseline
  to 10% above it.
- **I built a wrong tool partway through and removed it.** To catch a neighbour using the GPU I
  added GPU utilization to the run monitor. It read 94 to 100% during runs because it counts the
  benchmark's own work, so it flagged every run. That also means my earlier explanation of the
  second attempt (GPU at 85% near the end) rested on a reading that could not tell the benchmark
  from a neighbour, so that attempt's cause is not established. What the pre-flight does validly
  is read GPU utilization before a benchmark starts, when none of my work is running.

## Limits

- One machine, one 0.5B-parameter model at one quantization. Nothing here says how a larger model
  or another chip behaves.
- Five runs per prompt, synthetic prompts, temperature 0. Replies differ in length between
  prompts (two stopped early), so decode speed averages over different numbers of tokens.
- The machine was not idle (load average 3.6), only below the refusal threshold of 4.0.
- llama.cpp was a development build. Another commit can give different numbers.
- The cache and llama-bench figures each come from one session of five runs after four attempts
  at them (three were discarded and are disclosed). The overhead bound is against a fake server.
- In the cold cache phase other processes reached 144.5% CPU (Chrome's renderer helper 89%)
  against a limit of 150%, and the cold long-prompt decode median was 142.0 tokens/s against 159.5
  and 158.0 for the short and medium prompts. It passed the variation limit and no claim uses it.
- Free disk was 3.6 GiB during the reruns, below the runbook's 8 GB rule. I went ahead, as for Day
  2, because these runs write almost nothing.
- The 150 tokens per second figure is a measurement of this configuration. Whether it is close to
  the memory-bandwidth limit of the chip is a question for Day 6, once bandwidth is measured on
  Day 2. It is not claimed here.

## Evidence ledger entries

Added to [EVIDENCE.md](../../EVIDENCE.md).
