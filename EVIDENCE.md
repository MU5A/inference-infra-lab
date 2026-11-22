# Evidence ledger

Every claim this project makes, and what it rests on. The resume on Day 28 is written from this
file and from nothing else.

## Rules

- A claim enters the ledger the day it is made, with the file that supports it.
- **Resume-safe** requires label M or G, at least three runs, a documented method, and the
  hardware named in the claim.
- An S claim may appear on a resume only as "modelled" or "simulated". An L claim may appear only
  as "studied". Neither may be written as a result.
- If a later day contradicts a claim, the ledger row is corrected in place and the correction is
  noted. It is not quietly deleted.

## Ledger

| Day | Claim | Label | Evidence (file and command) | Numbers | Runs | Resume-safe |
|---|---|---|---|---|---|---|
| 1 | Qwen2.5-0.5B-Instruct Q4_K_M decodes at about 150 tokens/s on an Apple M3 with 16 GB, using llama.cpp commit cbb1c1c with Metal | M | `days/01-harness-and-baseline/results/baseline_llama_server.json` from `python -m inferlab bench` | median 152.6, 153.2, 149.4 tok/s for prompts of 40, 257, 1,469 tokens | 5 per prompt, after 2 warm-up | Yes, naming the chip, model and quantization |
| 1 | Time to first token with a cold prompt cache is 26.0 ms for a 40-token prompt, 92.9 ms for 257 and 490.8 ms for 1,469 | M | same file | medians; ranges in the day's README | 5 per prompt | Yes |
| 1 | Prompt processing speeds up with prompt length, from about 1,600 to about 3,000 tokens/s | M | same file, server-reported `prompt_per_second` | 1,592, 2,805, 3,005 | 5 per prompt | Yes |
| 1 | The harness's decode rate agrees with the server's own timer to within 0.1% | M | same file | 0.01% to 0.07% | 5 per prompt | Yes, as a statement about method |

## Built things, verifiable in the repository

These are engineering claims, not measurements. Each can be checked by running the command shown.

| Day | Claim | Check |
|---|---|---|
| 1 | A benchmark harness with a streaming client, TTFT, inter-token latency and decode-rate metrics, percentiles checked against NumPy, and an environment record | `pytest -q` (98 tests) |
| 1 | The harness refuses to benchmark a busy, throttled or battery-powered machine unless overridden, and records any override | `tests/test_cli.py`, `tests/test_runner_env.py` |
| 1 | The tests were checked by deliberately breaking the code 17 ways, and all 17 breakages fail the suite | `python scripts/mutation_check.py` |
