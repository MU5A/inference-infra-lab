# Day 1: benchmark harness and baseline

**Stage:** 1, systems foundations. **Label:** M (measured on this machine). **Date:** 2026-11-22.

## Goal

Build a measurement tool I can trust, then use it to take one baseline: how fast does a small
model run on this laptop, and how much of what the tool reports is the tool itself? Every later
day reuses this harness, so a mistake here would quietly corrupt every result after it.

## Environment

| | |
|---|---|
| Machine | Apple M3, 8 CPU cores, 10 GPU cores, 16 GB unified memory, macOS 26.2, arm64 |
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
   send time, and can fail on purpose, then 98 tests against it.
5. Checked that the tests can fail: 17 deliberate breakages of the code, each of which must turn
   the suite red. All 17 are caught. One was missed at first (see below).
6. Took the baseline with `python -m inferlab bench` on a clean machine.

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

## What failed or surprised me

- **Two assumptions about the server were wrong** and would have produced wrong numbers. The
  token count includes the end token, so the decode rate is computed from text events and the
  usage-based rate is kept beside it. And the server can reuse a cached prompt prefix between
  requests (every response carries a cached-token count), which would make repeated prompts look
  nearly free. I have not yet measured how large that effect is. Cold runs turn the cache off,
  and every result records the cached-token count, which is zero here.
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

## Still to do for Day 1

Three measurements were attempted on Day 2 and were disturbed by other activity on the machine (a
video player using the GPU, other processes raising the load). Their files are in
[`results/rejected/`](results/rejected/README.md) with the reason, and none of their numbers is
claimed anywhere. They will be repeated, with conditions verified throughout the run:

- The harness's own timing error, measured three times against a fake server with known timing.
- A cold versus warm prompt cache comparison. The effect is large (time to first token fell from
  tens or hundreds of milliseconds to about 10), but the exact numbers are not citable yet.
- A cross-check of decode speed against `llama-bench`.

What the attempts changed in the harness: the pre-flight now samples GPU utilization, and results
are judged by their own run-to-run variation, so a disturbed run is marked unstable whatever caused
it. See the log, section 10.

## Limits

- One machine, one 0.5B-parameter model at one quantization. Nothing here says how a larger model
  or another chip behaves.
- Five runs per prompt, synthetic prompts, temperature 0. Replies differ in length between
  prompts (two stopped early), so decode speed averages over different numbers of tokens.
- The machine was not idle (load average 3.6), only below the refusal threshold of 4.0.
- llama.cpp was a development build. Another commit can give different numbers.
- The 150 tokens per second figure is a measurement of this configuration. Whether it is close to
  the memory-bandwidth limit of the chip is a question for Day 6, once bandwidth is measured on
  Day 2. It is not claimed here.

## Evidence ledger entries

Added to [EVIDENCE.md](../../EVIDENCE.md).
