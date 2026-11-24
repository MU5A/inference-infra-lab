# Which Day 4 results may be cited

Each measurement phase was run three times as independent processes, judged against stability limits
fixed before the first run (10% variation), and kept on record whether or not it passed. Only the
rows marked **cite** are claimed anywhere.

| File | What it is | Status |
|---|---|---|
| `stream.json` | hand-written triad, scalar and float4, threadgroup sizes 32 to 1024, 256 MiB arrays | **cite**: stable (variation at most 7.2%), validated, every size ran as asked, no probe drift. The scalar rows have a wide run-to-run spread, which the day's README says. |
| `occupancy.json` | the scalar triad with 1 to 32 KiB of threadgroup memory declared per group of 256 | **cite**: the final attempt, stable (variation at most 2.6%) and validated. The sampler flagged it contended, over a window that includes the rests; see the day's README. |
| `fma.json` | fused multiply-add loop, fp32 and fp16, threadgroup sizes 32 to 1024 | **cite**: stable (variation at most 0.3%), validated, no probe drift |
| `rejected/occupancy_attempt1_unstable.json` | the same occupancy sweep, attempt 1 | do not cite |
| `analysis.json` | Day 3's MLX figures used for the comparisons, and the prediction verdicts | **cite**, as derived from the files above and Day 3's results |

`tables.md` is generated from these files by `scripts/day04_report.py`. The reason for the rejection
is in [`rejected/README.md`](rejected/README.md).
