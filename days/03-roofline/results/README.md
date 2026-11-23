# Which Day 3 results may be cited

Each measurement phase was run three times as independent processes, judged against stability limits
fixed before the first run (10% variation), and kept on record whether or not it passed. Only the
rows marked **cite** are claimed anywhere.

| File | What it is | Status |
|---|---|---|
| `gpu_bandwidth.json` | streaming kernels through MLX, 256 MiB arrays | **cite**: stable (variation at most 2.5%), validated, machine speed unchanged across the phase |
| `gpu_matmul.json` | square matrix multiply, fp32, fp16, bf16, n = 512 to 4096 | **cite**: stable (variation at most 2.8%), every product checked against NumPy, no drift |
| `gpu_skinny.json` | decode-shaped products, batch 1 to 1024 against an 8192 x 8192 fp16 weight matrix | **cite**: attempt 3, stable (variation at most 1.0%) and validated. The neighbour sampler flagged it contended, but its window includes the 240 s rests between runs, so the flag cannot say whether anything overlapped the timed work. See the day's README. |
| `rejected/gpu_skinny_attempt1_unstable.json` | the same sweep, attempt 1 | do not cite |
| `rejected/gpu_skinny_attempt2_unstable.json` | the same sweep, attempt 2 | do not cite |
| `analysis.json` | roofs, placements and verdicts computed from the three files above | **cite**, as derived from them |
| `specs_table.md` | vendor specification figures, label L | **cite** as studied, with the dates and sources in `../specs/hardware_specs.json` |

`measured_tables.md`, `specs_table.md` and `roofline.svg` are generated from these files by
`scripts/day03_report.py`. The reasons for each rejection are in [`rejected/README.md`](rejected/README.md).
