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
