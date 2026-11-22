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
| 2 | In a 96 MiB, no-swap Linux container (Docker Desktop VM, kernel 7.0.12-linuxkit), a process allocating ordinary memory was OOM-killed after 72 MiB | M | `days/02-memory-hierarchy/results/linux_memory.json` from `python scripts/day02_linux.py` | 72 MiB in every run | 3 containers | Yes, as behaviour, naming the environment |
| 2 | A memory-mapped, read-only 192 MiB file (twice the limit) could be read completely in that container with total memory held at the limit (95.89 to 95.97 MiB) and no OOM kill | M | same file | limit reached 117 to 125 times | 3 containers | Yes |
| 2 | Writing to a privately mapped file in that container was OOM-killed after 80 MiB | M | same file | 80 MiB in every run | 3 containers | Yes |
| 2 | Writing a 192 MiB file without fsync was OOM-killed in a 96 MiB container, and the identical write with fsync after each chunk completed | M | same file, `dirty_write` | 3 of 3 pairs | 3 pairs | Yes. The dirty-page mechanism was not directly observed. |
| 2 | Sustained STREAM-style triad memory bandwidth on the Apple M3 was 80.3 to 86.1 GB/s for every thread count from 1 to 8 | M | `days/02-memory-hierarchy/results/microbench.json`, `microbench/membw.c` | medians 81.3 to 85.1 GB/s, variation at most 1.6% | 3 runs per thread count | Yes, naming the chip. Which cores ran is not known. |
| 2 | Load-to-use latency on the M3's performance cores was 1.1 ns up to 128 KiB, 4.6 to about 8 ns from 256 KiB to 11 MiB, and 103 ns for main memory | M | same file, `microbench/chase.c` | plateau medians 1.11, 5.0, 7.1 and 103.4 ns, variation 1.9%, 6.9%, 9.6% and 0.6% | 3 sweeps | Yes. The 128 KiB edge matches the OS-reported L1D. The middle boundaries did not repeat and are not claimed. |
| 2 | Single-precision matrix multiply on the M3: a reordered loop is about 10 to 14 times faster than the textbook loop (about 30 against 2.2 to 3.6 GFLOPS), my tiled versions were slower than the reordered loop, and Apple's Accelerate reached 1.3 to 1.65 TFLOPS | M | `days/02-memory-hierarchy/results/compute.json` | all 24 entries within 1.5% variation | 3 runs, best of 8 each, after 3 earlier attempts | Yes, with the attempts disclosed |
| 2 | Reading a 469 MB file cold with a sequential read ran at about 2.4 GB/s, and reading it by touching memory-mapped pages was 11 to 25 times slower in every trial | M | `days/02-memory-hierarchy/results/microbench_rerun.json` | read 2.32 to 2.48 GB/s (variation 2.8%); touch 0.10 to 0.22 GB/s (unstable, so the ratio only) | 3 trials, each on a fresh cold copy | Yes for the ratio and the read speed. Not the touch speed. |

## Built things, verifiable in the repository

These are engineering claims, not measurements. Each can be checked by running the command shown.

| Day | Claim | Check |
|---|---|---|
| 1 | A benchmark harness with a streaming client, TTFT, inter-token latency and decode-rate metrics, percentiles checked against NumPy, and an environment record | `pytest -q` (98 tests) |
| 1 | The harness refuses to benchmark a busy, throttled or battery-powered machine unless overridden, and records any override | `tests/test_cli.py`, `tests/test_runner_env.py` |
| 1 | The tests were checked by deliberately breaking the code, and every breakage fails the suite | `python scripts/mutation_check.py` |
| 1 | The pre-flight refuses a busy GPU, low free memory, battery power and CPU load, and results are judged by their own run-to-run variation | `tests/test_runner_env.py` |
| 2 | Five C microbenchmarks that validate their own results, compile and pass on macOS and Linux, and a plateau finder that reads cache levels off a latency curve | `pytest -q`, `microbench/` |
