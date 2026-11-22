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

## 8. Not done yet

Three measurements need a quiet machine and were refused by the check. They are listed in
[README.md](README.md) as not yet measured.
