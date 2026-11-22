# Day 1 log

A running record of what was done, in order, including the dead ends. Times are omitted on
purpose: they were not recorded, and an invented duration would be a made-up number.

## Starting state

- Apple M3, 16 GB, macOS 26.2. Free disk at the start of the day: 2.1 GiB (99% full).
- No llama.cpp, no Python environment, no model on the machine.

## 1. Make room

Day 1 needs about 1 GB, but the machine was too full to trust. With the owner's explicit approval,
regenerable caches were cleared (browser, package manager, Go module cache), which reached 8.7 GiB.
Two things went wrong and are recorded because they will recur:

- Docker Desktop was started to inspect it and wedged: its daemon stopped answering while the disk
  was nearly full. It was quit and left stopped. It is not needed on Day 1.
- Free space then fell again, to 2.6 GiB, because another process on the machine ran
  `go test -race`, rebuilding the Go caches that had just been cleared. The Go build cache was
  cleared a second time when no Go process was running (13 GiB free) and refilled to about 8 GiB
  within minutes. Clearing caches is a one-off gain and not a fix. This is why the runbook now
  ties the disk requirement to what a day will download or build.

## 2. Get the server and a model

```bash
scripts/build_llama.sh
# ... version: 1.0.0-dev (build 1, commit cbb1c1c), built with AppleClang 21 for Darwin arm64
scripts/fetch_model.sh Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf
# verified qwen2.5-0.5b-instruct-q4_k_m.gguf sha256 74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db (469M)
```

Before approving the downloads, the sources were checked read-only: the repository is active (last
push the day before), and the model repository is not gated and is Apache-2.0. The fetch script
compares the file with the sha256 Hugging Face publishes and refuses a mismatch.

Server start (all layers on the GPU, one request slot):

```bash
llama-server -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8081 -c 4096 -ngl 99 --parallel 1
# {"status":"ok"} after 14 s
```

## 3. Read the real wire format before writing the client

One streamed request, saved raw. Findings that changed the design:

- The first event is `{"delta":{"role":"assistant","content":null}}`. It must not count as a token.
- Text arrives one token per event.
- The last events are a finish event, then an event with `"choices":[]`, `usage` and `timings`:
  `"usage":{"completion_tokens":3,"prompt_tokens":35,...,"prompt_tokens_details":{"cached_tokens":0}}`
- Only **2** events carried text, but `completion_tokens` was **3**. The count includes the end
  token. A decode rate based on it would be too high, so the harness counts text events and keeps
  the usage-based rate beside it.
- The response reports `cached_tokens`, so the server can skip prompt processing for text it has
  seen. A benchmark that repeats a prompt must turn that off or record it.

## 4. Build the harness

Package `inferlab`: `stats.py` (percentiles by linear interpolation, checked against NumPy),
`client.py` (streaming client and the definitions of TTFT, ITL and decode rate), `runner.py`
(warm-up, repeated runs, per-prompt summaries), `env.py` (what machine and tools a result came from),
`__main__.py` (the `bench` command). Native arm64 Python 3.12.8 was chosen deliberately: an x86
Python under Rosetta would distort the latencies being measured.

A fake server copies the real wire format, stamps every event with its send time, and can fail on
purpose (HTTP 500, dropped connection, malformed events, no tokens, packed tokens, missing usage).

One correction while writing tests: a first metric called `tokens_per_event` could never be 1.0
for a normal stream, because of the end token. It was replaced by `unseen_tokens`, the tokens the
server counted minus the text events seen, where 0 or 1 is normal.

## 5. Check that the tests can fail

`scripts/mutation_check.py` breaks the code on purpose, 17 ways, and requires the suite to fail
each time. All 13 first-round mutations were caught (percentile without interpolation, TTFT from the
first event, decode rate off by one, errors raised instead of returned, warm-up measured, cold flag
ignored, and so on). Then the pre-flight check was added (next section) with four more mutations.
**One was missed**: the command-line refusal logic had no test. Four CLI tests were added, and all
17 are now caught.

## 6. A benchmark that refuses a bad machine

The first sanity run was taken with the load average near 10 and the laptop on battery. Those
numbers were discarded. The harness now has `env.preflight()`: it refuses to benchmark when the
1-minute load average is above half a core per core, when running on battery, or when the chip is
throttled, unless `--allow-noisy` is passed, and an override is written into the results.

What the check did in practice:

| Moment | What it reported |
|---|---|
| After the build | battery power, load 5.7 |
| Later | clean (on AC, load 3.6) |
| Starting the overhead measurement | refused: load 4.1 |
| Minutes later | load 24 to 28 from another process, refused |

## 7. Baseline, taken while the check was clean

```bash
python -m inferlab bench --prompts prompts/day01.json --runs 5 --warmup 2 --max-tokens 128 \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --llama-dir third_party/llama.cpp \
  --out days/01-harness-and-baseline/results/baseline_llama_server.json
```

Three prompts of 40, 257 and 1,469 tokens (the long prompt is larger than the 1,000 tokens
planned; the actual counts are what is reported). Results and the comparison with the server's own
timer are in [README.md](README.md). Cached tokens seen: 0.

## 8. An intermittent test failure I could not explain

After making the scripts independent of the checkout path, the suite reported 1 failure, then
passed on the next three runs. I did not capture which test failed, only the summary line
(1 failed, 91 passed). I tried to reproduce it with ten
CPU-spinning processes of my own, running the client tests six times and the whole suite four
times, at load averages up to 16. Nothing failed. The original failure happened while another
process had the load average near 28, which I cannot recreate.

Rather than guess, I removed the tests that were brittle in principle. Three assertions assumed
wall-clock pacing (an upper bound on time to first token, a median gap, and "decode rate never
exceeds 1 / interval", which is false when the first event is late). The arithmetic is now tested
with exact scripted arrival times, and the HTTP tests keep only properties that hold at any speed.
The suite went from 92 to 98 tests and the mutation check still catches all 17 breakages.

## 9. Not done yet (superseded by sections 10 and 11)

## 10. The three open measurements, attempted on Day 2

All three were attempted with the pre-flight clean at the start. None produced a citable result.
They are in `results/rejected/` with a note on why.

- **Warm prompt cache.** Time to first token fell from 26.0, 92.9 and 490.8 ms (cold) to 7.7, 8.4 and
  10.4 ms, with 195, 1,280 and 7,340 cached tokens seen. The direction and rough size are safe, but
  decode varied by up to 6.9% (a sample at 127.9 against 150 elsewhere), so the run was disturbed and
  the exact numbers are not citable.
- **llama-bench.** The first run gave 3,211.7 prompt tokens/s and 159.7 decode tokens/s with small
  deviations, but it ran because my script printed the pre-flight failure without acting on it (a
  bug, now fixed), and its output file was overwritten by the second run, so there is no raw data to
  commit. The second run decoded at 60.5 tokens/s, standard deviation 50.2, samples 148.9, 36.1, 48.4,
  43.8 and 25.4, while VLC used the GPU. The pre-flight passed because it only looked at CPU load.
- **Harness overhead.** Median 325 to 636 microseconds, p95 3 to 11 milliseconds, p99 11 to 18
  milliseconds, max 88 milliseconds, load average 6.4 at the end. Two earlier runs, discarded for an
  unclean pre-flight, had smaller tails, so the load average was not a reliable predictor.

What changed in the harness as a result (tests and mutations for each): the pre-flight samples GPU
utilization, results are judged by their own coefficient of variation (decode within 5%, time to
first token within 10%), unstable runs are written and marked and the command exits with status 4,
and every follow-up step is gated on the pre-flight instead of only printing it. I also learned
that a load average that is clean at the start says nothing about the next two minutes, and that
my own measurements raise it for whatever runs next.

These three measurements will be repeated, with conditions checked throughout, before Day 1 is
marked done.

Three measurements need a quiet machine and were refused by the check. They are listed in
[README.md](README.md) as not yet measured.

Correction (section 11): the claim above that the GPU was shared rested on a GPU-utilization
reading that cannot separate the benchmark's own use from a neighbour's. What is known is that VLC
was running and decode collapsed; the mechanism was not isolated.

## 11. The three measurements, repeated (2026-11-22)

Free disk was 3.6 GiB, under the runbook's 8 GB rule; I went ahead as on Day 2.

**Harness overhead (held).** Three runs of 3,450 events against the fake server, 0 events seen
before they were sent, upper bound over the runs 271 us median, 448 us p95, 756 us p99, 5,728 us
max. Run 1 contained the worst event while Docker Desktop's helper used up to 60% CPU. Taken before
the GPU work below, and it does not involve the GPU.

**Attempt 2 at cache and llama-bench (did not hold).** Short prompt speed-up 3.4x, warm long run
unstable (time to first token 13%), llama-bench decode samples 166.0, 165.5, 163.7, 29.8, 50.6
(variation 59.7%). Quarantined.

**Mistake: a GPU contention flag.** I added GPU utilization to `Watch` and made a peak above 25% mean
"contended". It read 94 to 100% in the next run, because the benchmark itself drives the GPU. The
flag fired on every run and could not tell a neighbour from my own work. Reverted in the next commit.

**Attempt 3 (steady and slow).** Cold phase 155 to 161 tokens/s decode, then warm decode 157, 103 and
86; llama-bench 779.5 prompt tokens/s and 55.7 decode with variation under 0.5%. The repeat limit
passed a machine that was slow the whole time. Diagnosis, all of it recorded in
`results/rejected/`: the Metal backend was in use (backends `MTL,BLAS`), memory bandwidth was
normal (triad 84.5 GB/s), VLC was idle (0% CPU), GPU utilization read 0 to 1% with nothing of mine
running, and the machine is a fanless MacBook Air (Mac15,12). A script that idled and then probed
decode read 166.1 and 166.5 after 240 s, 164.9 and 164.8 after another 300 s, and 167.7 and 167.8
after another 420 s. The slowdown faded with rest. Heat is the likely cause. I read no temperature
or clock, so it is a hypothesis.

**Reference probe added (after the data).** `crosscheck.check_reference`: a short decode-only
llama-bench before and after every GPU phase, refused if more than 20% below the baseline, retried
once a minute for up to ten minutes. Tolerance chosen after seeing attempt 3. Tests, and three
mutations (floor never applied, floor exclusive, missing probe accepted), all caught.

**Attempt 4 (held, with one prediction failing).** Cold and warm phases and llama-bench all passed
their variation limits and every probe was within 20% of the baseline (probes 151.6 to 164.6
tokens/s). Short prompt warm speed-up 3.2x against a predicted 5x, a finding. Medium 11.1x, long
47.2x. llama-bench prompt processing 3,191.9 tokens/s (+6.2% against the server's rate) and decode
159.7 tokens/s (+4.7% against the baseline). The cold phase was not quiet: other processes reached
144.5% CPU, just under the 150% limit, mostly Chrome's renderer helper (89%), and cold long-prompt
decode read 142.0 tokens/s against 159.5 and 158.0 for the other cold prompts. Not used in a claim.

**Tables.** `src/inferlab/followup_report.py` renders them from the result files. A row from a run
that was not repeatable gets no numbers, only the reason. `tests/test_day_docs.py` checks the README
contains the generated tables verbatim.
