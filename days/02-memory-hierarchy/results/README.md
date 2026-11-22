# Which Day 2 results may be cited

Each measurement group was run, judged against stability limits fixed before the first run, and
kept on record whether or not it passed. Only the rows marked **cite** are claimed anywhere.

| File | Group | Status |
|---|---|---|
| `microbench.json` | bandwidth | **cite**: stable, variation at most 1.6% |
| `microbench.json` | latency, performance cores | **cite**: same plateaus in all 3 sweeps |
| `microbench.json` | latency, efficiency-core request | do not cite: the sweeps found 6, 4 and 7 plateaus |
| `microbench.json` | compute | do not cite: attempt 1, disturbed (19 of 24 unstable) |
| `microbench.json` | cold read | do not cite: first design, had no sequential-read control |
| `microbench_rerun.json` | cold read | **cite** the sequential read and the ratio. The memory-mapped touch and WILLNEED rows are unstable and are given as ranges only. |
| `microbench_rerun.json` | compute | do not cite: attempt 2, n=128 too short to time |
| `compute_attempt3.json` | compute | do not cite: attempt 3, disturbed by another process's builds |
| `compute.json` | compute | **cite**: attempt 4, stable on all 24 entries |
| `linux_memory.json` | Linux containers | **cite**: every prediction held in all runs |

The `_table.md` files are generated from these by `scripts/day02_report.py`.
