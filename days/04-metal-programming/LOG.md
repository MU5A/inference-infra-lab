# Day 4 log: GPU programming concepts on Metal

Appended as the work went. Times are local (WAT). This day was run by the scheduled task
`inference-infra-lab-daily`, with nobody watching.

## 1. Start, 2026-11-24 09:37

- Pre-flight (DAILY.md section 1): free disk 8.6 GiB (`df -h .`, at least 8 GB), `gh auth status`
  logged in as MU5A, git tree clean at 7929246 and level with origin/main, on AC power (battery
  85%, charging). The day downloads nothing: MLX 0.32.4 was installed on Day 3 and is reused.
- Machine: MacBook Air (Apple M3, `system_profiler` reports a 10-core GPU, Metal 4), macOS 26.2,
  Python 3.12.8, MLX 0.32.4.
- The machine was busy at the start: one-minute load 16.3, with Slack (about 90% of a core),
  `ANECompilerService` and `IntelligencePlatformComputeService` (about 47% each) the busiest. The
  runner waits, bounded, for the pre-flight to pass before each phase.

## 2. Predictions first

`src/inferlab/metal_sweep.py` holds the five predictions and the checks that judge them, with
tests on synthetic data (`tests/test_metal_sweep.py`). It was committed before any kernel was
written or run, so nothing measured today could have shaped it. The two Day 3 numbers it leans on
(MLX triad 87.2 GB/s, MLX fp32 matrix multiply 2.78 TFLOPS) are read from Day 3's result files at
judging time, not retyped.

## 3. Kernels, worker and runner

- `scripts/day04_worker.py`: three hand-written Metal kernels through `mx.fast.metal_kernel`
  (triad scalar and float4, triad with a threadgroup-memory exchange, an 8-chain FMA loop). Thread 0
  of each writes the threadgroup size it ran with and `threads_per_simdgroup`.
- `scripts/day04_run.py`: Day 3's gated protocol (quiet wait, decode probe before and after,
  neighbour sampler, three process runs, 10% limit).
- Smoke run, 09:50: `python scripts/day04_worker.py --phase P --smoke` for each phase, on 64 Ki
  elements, prints validation and dispatch only, no timing. Every one of the 30 configurations
  validated and ran at the requested threadgroup size (32 to 1024), and every kernel reported a
  SIMD-group width of 32. Unlike Day 3's dry run, no rate was printed or seen before the gated runs.
  Template integers (`SCRATCH`, `ITERS`) and a threadgroup array declared in the kernel body both
  work in MLX 0.32.4's `metal_kernel`; I had not been sure they would.

## 4. Protocol for the gated runs, fixed before the first one

`python scripts/day04_run.py --rest 120 --wait-quiet 900`. The 120 s rest before each process run
applies to all three phases from the start, because Day 3's only unstable phase was its sustained
compute sweep and it held once rests were added. The 10% limit and the 20% decode-probe limit are
unchanged from Day 3.

## 5. Attempt 1, all three phases, 09:42 to 10:07

`python scripts/day04_run.py --rest 120 --wait-quiet 900`, exit 4.

- stream: held. Stable, validated, every configuration ran at the threadgroup size asked. Decode
  probe 155.6 before, 155.8 after.
- occupancy: did not hold. 24 KiB varied 11.4% (runs 93.9, 72.8 and 93.7 GB/s, the second slow)
  against the 10% limit; the other five sizes were within 3.7%. Probe 153.7 before, 160.8 after,
  no drift. The file is kept as `results/rejected/occupancy_attempt1_unstable.json` and none of its
  numbers is cited.
- fma: held. Every configuration within 0.2%. Probe 161.9 before, 155.1 after.
- The neighbour sampler flagged all three phases contended (peaks 260% to 355% of a core for all
  other processes; `node`, `terraform`, `bird`, `duetexpertd`, `ANECompilerService` among the
  busiest). As on Day 3 the sampler window includes the 120 s rests, so the flag cannot say whether
  a burst overlapped the timed work. The run-to-run variation decides stability.

Decision: re-run the occupancy phase once with the same protocol, unchanged. If it does not hold,
the day stops with STUCK.md. I have now seen attempt 1's occupancy numbers, so attempt 2 is not
blind; the predictions were committed in c11afda before any run.

## 6. The session lost its network, and a correction, 2026-11-24 about 12:00

The scheduled session ended at 10:49 with `API Error: Can't reach the API server (ENOTFOUND)`,
after `STUCK.md` was written at 10:39. `STUCK.md` says the `stream` and `fma` result files are
committed. They were not: git showed `results/` untracked at 12:00, because the session ended
before that commit. Checked against the files at that point: `stream` stable (variation at most
7.2%), validated, no drift; `fma` stable (at most 0.2%), validated, no drift; `occupancy` attempt
1 rejected at 11.4%, as recorded in section 5. They are committed now, with `cuda_mapping.md` and
this log, locally only. Nothing is pushed. The Mac was on AC at 09:37 and on battery by 10:39,
which is what stopped the second occupancy attempt. This log does not record a full run of
`scripts/mutation_check.py` for the Day 4 checks, so that run is still to do.
