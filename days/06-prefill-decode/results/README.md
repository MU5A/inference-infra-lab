# Which Day 6 results may be cited

The bench phase ran every model once per round, three rounds, each run its own `llama-bench`
process bracketed by Day 1's decode probe. The time-to-first-token phase ran one `llama-server`
per model with a warm-up and three requests per prompt. No probe was refused, no request failed,
and no attempt was set aside, so there is no `rejected/` folder today. A figure is cited only if
its three runs agree with a coefficient of variation of at most 10%.

| File | What it is | Status |
|---|---|---|
| `gguf/*.json` | the header of each of the four model files (tensors, types, sizes, metadata) and its sha256, from `scripts/day06_run.py --phase headers` | **cite** |
| `router.json` | the router simulation, from `--phase router` | **cite**, as simulation (S) |
| `bench.json` | the 12 llama-bench processes: raw entries, per-process rates, probe before and after, CPU neighbours, machine state before and after, environment | **cite**, except the 0.5B Q4_K_M decode at depth 4,096 (`tg@4096`), which varied by 15.5% and is cited nowhere |
| `ttft.json` | time to first token for the 40, 257 and 1,469-token prompts on each model, with probes and neighbours | **cite** |
| `analysis.json` | the verdict on each prediction and every number in the tables, recomputed from the files above and Day 3's bandwidth by a test | **cite** |
| `tables.md` | rendered from `analysis.json` by `scripts/day06_report.py`; the write-up embeds it verbatim | **cite** |
