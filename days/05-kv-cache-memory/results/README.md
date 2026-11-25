# Which Day 5 results may be cited

The llama.cpp matrix is 12 configurations (four contexts, three cache types) launched three times
each, as independent server processes. Every launch had to start, answer `/health`, stop with status
0 after SIGINT, log exactly one KV cache line, print a memory breakdown at shutdown and report flash
attention on, or it was marked unusable. All 36 passed. Only the rows marked **cite** are claimed
anywhere.

| File | What it is | Status |
|---|---|---|
| `llama_kv.json` | the 36 launches of `scripts/day05_run.py`, with the parsed KV line, parameter count, buffers and shutdown breakdown of each, and the machine state before and after | **cite**: all 36 usable, each configuration identical across its three launches |
| `gguf_summary.json` | the model file's header (291 tensors, shapes, types, offsets), read three times with identical results, from `scripts/day05_gguf.py` | **cite** |
| `analysis.json` | the verdict on each of the ten predictions and every number the tables are built from | **cite**, as derived from the files above, `../published.json` and `../configs/` |
| `tables.md` | generated from `analysis.json` by `scripts/day05_report.py`; the write-up embeds it verbatim | **cite** |
| `sample_server_log.jsonl` | one complete server log (4,096 tokens, f16 cache, first launch), kept so the parsing can be re-checked | evidence for the parser, not a source of new numbers |
| `rejected/llama_kv_attempt1_parser.json` | the first complete run, set aside because my parser rejected it | do not cite |

`../published.json` holds the figures read from model cards and papers on 2026-11-25 (label L),
with the page each came from. `../configs/` holds the five pinned model configuration files. The
reason for the rejection is in [`rejected/README.md`](rejected/README.md).
