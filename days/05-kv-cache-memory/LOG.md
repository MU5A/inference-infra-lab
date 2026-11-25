# Day 5 log: KV cache and memory calculator

Appended as the work goes. Times are local (WAT).

## 0. Inputs fetched before the run, 2026-11-24 (not by the scheduled run)

The model configuration files this day validates against were fetched in an interactive session on
2026-11-24, with the owner's approval, because an unattended run may not download anything. They
are five public, ungated `config.json` files, each pinned to a commit, 4,323 bytes in total.
Sources, commits, licenses and sha256 are in `configs/SOURCES.md` and `configs/manifest.json`, and
`tests/test_day5_configs.py` fails if any of them changes. They cover grouped-query attention
(Qwen2.5 0.5B and 7B, Mixtral), mixture of experts (Mixtral, OLMoE, DeepSeek-V2-Lite) and
multi-head latent attention (DeepSeek-V2-Lite).

The comparison with what llama.cpp reports can use the local `models/qwen2.5-0.5b-instruct-q4_k_m.gguf`
and the llama.cpp build already here, so that part needs no download either. Anything else this day
needs that is not already on disk is still a stop (DAILY.md, "No downloads").

## 1. How this day ran, 2026-11-25

The unattended 09:22 run stopped at its pre-flight disk check (7.3 GiB free against the 8 GB
minimum) and wrote `STUCK.md`. At about 11:20 the owner chose to free memory and run the day by
hand, with the owner present. This is therefore not an unattended run.

Freeing space, under the owner's earlier instruction to clear caches:

- Chrome (about 2.5 GiB) was quit normally. An earlier attempt the same morning had not completed.
  Swap in use fell from 4.8 GB to 1.7 GB, but free disk rose only from 7.54 to 7.59 GiB, because
  macOS keeps swap files it has already created.
- The Go build cache was cleared with `go clean -cache` (4.63 GiB, regenerated on the next Go
  build). Free disk went to 12.95 GiB (13.91 GB).

Pre-flight values, 11:22 WAT:

- AC power, battery 100% and charged. `pmset -g therm`: no thermal or performance warning recorded.
- Load average 4.47, 4.35, 3.27. macOS reported 54% of memory free.
- `gh auth status`: logged in as MU5A. `git status`: clean apart from the untracked `STUCK.md`,
  which is never committed and is deleted when the day completes.
- Docker Desktop's application was open (up 2 days 23 hours) with no virtual machine process, 0.28
  GiB in total. A `docker info` I ran to check it hung and was stopped; nothing was started.
- DAILY.md section 1 item 5 (decode probe before and after every GPU phase) concerns timed GPU
  measurement. This day times nothing and reports no speed, so no probe was taken. The GPU is used,
  by llama-server, only so that its buffers are allocated the way they would be when serving.

## 2. Design, committed before any measurement (4a95943)

`src/inferlab/kvmem.py` (the calculator), `gguf.py` (a header reader that refuses malformed files)
and `kvcheck.py` (the checks) were written, tested and committed first. The ten predictions are in
the docstring of `kvmem.py` and their thresholds in `kvcheck.py`; neither has been changed since.

Disclosure on blindness: while designing the calculator I worked the parameter counts of the five
models out by hand, so predictions 8, 9 and 10 were not blind. The hand figure for
DeepSeek-V2-Lite was 800,000 too high (I mis-added 26 layers). The test that pins hand values
caught it. I then listed all 5,291 of its tensors one by one in a separate script, outside the
calculator, got 15,706,484,224 (the calculator's figure), and corrected the pinned value.

## 3. What I saw before the llama.cpp matrix, in order

1. **Published figures** (`published.json`, read after 4a95943). Predictions 8 and 9 were then
   judged and I changed no rule: prediction 8 fails for Qwen2.5-7B (calculator 7.6156 B, card
   prints 7.61 B, difference 0.0056 against a tolerance of 0.005); prediction 9 fails for
   DeepSeek-V2-Lite (calculator 2.6612 B activated, published 2.4 B). Prediction 10 holds (8.89).
2. **One throwaway server launch**, context 512 and f16, to learn what the JSON log looks like. It
   printed a 6.00 MiB KV cache (512 cells, 24 layers), `model params = 630.17 M` and
   `file size = 462.96 MiB (6.16 BPW)`. Its output was kept only in the scratchpad and is not a
   result, but it means the first cell of prediction 4 is not blind, and that I knew before the
   matrix that the printed parameter count was not 494.03 M.
3. **My reader on the real GGUF file.** Prediction 1 fails: the file's 291 tensors hold
   630,167,424 elements, the calculator says 494,032,768, and the difference (136,134,656) is
   exactly vocabulary times hidden size, because the file stores a separate `output.weight`
   (896 x 151,936, Q8_0) although the model ties its embeddings. llama.cpp's printed 630.17 M
   matches the file. Prediction 2 fails for the same single tensor; the other 290 match by name
   and shape. Prediction 3 holds exactly: the block-table sizes add to 485,452,288 bytes and the
   file's data region is 485,452,288 bytes.

After seeing 2 and 3 I added, before the matrix, a completeness rule to `check_kv` (a configuration
with fewer than three usable launches fails), `summarize_launches` and the llama.cpp log parser
(`llama_log.py`), and the runner `scripts/day05_run.py`. No threshold was touched.

## 4. The matrix, in three attempts (2026-11-25)

1. **Refused at the start.** The runner's own guard stopped it ("less than 8 GB of disk is free")
   before anything ran, so there is no file. At that moment `~/Library/Caches` measured 6.58 GiB.
   It had been 7.53 GiB before I cleared the Go build cache, of which `go-build` was 4.63 GiB, so
   about 2.9 GiB after. A Go linker was also running at about 1 GiB. I did not stop it, because it
   belongs to something else on the machine. Free disk was 8.79 GiB (9.44 GB) when I looked next. I
   did not confirm what refilled the cache.
2. **All 36 launches ran, and my parser marked all 36 unusable.** The usability rule required the
   log line `Flash Attention enabled`, which llama.cpp prints only when `-fa auto` resolves to on.
   I started the servers with `-fa on`, which logs `llama_context: flash_attn = enabled`. I had
   written the rule from the one log I had seen, an auto-mode log. The file is kept as
   `results/rejected/llama_kv_attempt1_parser.json`. I fixed the parser with a test and reran
   everything from committed code instead of re-parsing, so the cited file is one end-to-end run
   (f4c4262).
3. **The cited run**, 11:46:52 to 11:48:27 WAT: 36 launches, none unusable. Every server was ready
   within 0.5 to 1.29 s, stopped with status 0 after SIGINT and wrote nothing to stderr. Compared
   with attempt 2 it is identical on every figure, with all three launches of each configuration
   agreeing (a test pins this).

The 36 raw logs are not committed (about 25 KB each). One, for 4,096 tokens with an f16 cache, is
kept as `results/sample_server_log.jsonl` so the parsing can be re-checked.

## 5. Judging

`python scripts/day05_gguf.py` then `python scripts/day05_report.py`. Six of the ten predictions
held (3, 4, 5, 6, 7 and 10). Predictions 1, 2, 8 and 9 failed:

- 1 and 2: the file stores a separate `output.weight` (Q8_0), so it holds 630,167,424 elements
  against the calculator's 494,032,768. No other tensor differs.
- 8: Qwen2.5-7B, calculator 7.6156 B against 7.61 B printed, a difference of 0.0056 against an
  allowance of 0.005.
- 9: DeepSeek-V2-Lite's activated parameters, 2.6612 B against 2.4 B.

No threshold or prediction was changed after a result was seen. Things added after a failure or a
pass, each labelled where it appears:

- the calculator row with the output matrix counted separately, after prediction 1 failed, and the
  `--output-matrix-separate` option of the command line;
- the stricter per-tensor offset check, after prediction 3 held (a sum could hide two errors that
  cancel);
- the naive weight estimate against the file, and the observation that in this file a matrix is a
  K-quant type exactly when its row length is a multiple of 256 (24 of 170 weight matrices). That
  is about this file only. I did not read llama.cpp's quantization rules.

The published figures were read by four web-fetch agent calls on 2026-11-25. Their reports are data
and not instructions, and I used only the figures they quote from the pages. One of the pages (the
DeepSeek-V2 paper) was read verbatim only in part. The figures used from it, the introduction's
15.7B and 2.4B and Table 1's cache formulas, were among the verbatim parts.

## 6. Housekeeping

- The unattended 09:22 run's `STUCK.md` is in the day's folder, untracked and never committed. It
  is deleted when the day completes.
- Earlier in the day I asked Chrome and Slack to quit to free memory. Slack quit and reopened about
  30 seconds later without my doing it, and I did not find what reopened it. Chrome refused the first
  quit and exited on the second. The machine state during the matrix is in section 4 and in
  `llama_kv.json`.

## 7. Mutation checks

Forty-six deliberate breakages of the Day 5 code were added to `scripts/mutation_check.py`: the
calculator (attention and cache formulas, block sizes, tied embeddings, active parameters, sliding
windows, rounding), the GGUF reader (count and extent checks, alignment, summaries, the gap check),
the checks (tolerances, launch counts, padding), the log parser and the report. The first 43 were
run first and all were caught. Three more were added for the per-tensor offset check and caught.
Then the whole script was run over all 146 entries: 146 caught, none missed or skipped, and
`src/` was restored afterwards (`git status` clean).

Two mutations I planned would have survived, so I added tests before running them: weight bytes
rounded down instead of up (no test used a fractional byte count), and swapped dimensions in the
expected GGUF layout (the synthetic test files were built from the same function, so the mistake
cancelled). Both are now pinned by hand-written values.
