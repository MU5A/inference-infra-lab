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
