# Rejected attempts, Day 5

Nothing in this folder is cited by the write-up.

## llama_kv_attempt1_parser.json

The first run of `scripts/day05_run.py` (2026-11-25, committed runner cf9692d). All 36 launches ran
(12 configurations, 3 repeats each) and every server stopped with status 0 after SIGINT, but the
file marks all 36 as unusable. My usability rule required the log line `Flash Attention enabled`,
which llama.cpp prints only when `-fa auto` resolves to on. I started the servers with `-fa on`,
which prints `llama_context: flash_attn = enabled` instead. I had written the rule from the one
log I had seen, which was an auto-mode log.

The fault was in my parser, not in the measurement. It was fixed, with a test
(`tests/test_llama_log.py`), and the whole matrix was rerun from committed code rather than
re-parsing these logs, so the cited file comes from one end-to-end run. No threshold or prediction
changed.

Its numbers are identical to the cited run's on every figure: for each of the 12 configurations,
the cell count, the K, V and total KV size and the model, context and compute columns of the
shutdown breakdown, with all three launches of each configuration agreeing. A test pins this
(`tests/test_day5_results.py`). It is kept out of the cited set because it failed its own usability
check, not because its figures differ, and a figure is cited from a run that passed.

An earlier attempt the same day produced no file at all. The runner refused to start because free
disk had fallen below 8 GB, and it ran nothing.
