# Day 4: GPU programming concepts on Metal

**Stage:** 2, GPU architecture and modern silicon. **Label:** M (measured on this machine) and L
(NVIDIA's CUDA guide, read, nothing run). **Date:** 2026-11-24. **Time spent:** elapsed from 09:37,
with long waits. The first part ran as a scheduled task with nobody watching; its session lost its
network at 10:49 and I finished the day with you present. Hands-on time was not recorded.

## Goal

Run three small Metal kernels that I wrote by hand through MLX, sweep how they are launched
(threadgroup size, and how much threadgroup memory each group declares), and say which CUDA ideas
(warp, thread block, shared memory, occupancy, tensor core) each result does or does not speak to.
Every CUDA statement is label L and was checked against NVIDIA's pages. Nothing here measures a
CUDA GPU.

How the results were judged: five predictions were written in `src/inferlab/metal_sweep.py` and
committed (c11afda) before any kernel existed or ran. Each phase ran three times as independent
processes and had to repeat within 10% (fixed before the first run), bracketed by a decode probe that
must be within 20% of Day 1's baseline. A phase that did not repeat was kept, not cited. Which files
may be cited is in [`results/README.md`](results/README.md).

## Environment

- MacBook Air (Mac15,12): Apple M3, 4 performance and 4 efficiency cores, 10-core GPU reported by
  `system_profiler`, 16 GB, no fan. macOS 26.2, Python 3.12.8, MLX 0.32.4 (installed on Day 3, so
  nothing was downloaded today).
- On AC power at 09:37 and again for the final measurement at about 12:19. It was on battery at 10:39,
  which is what stopped the second try at the occupancy phase.
- Free disk was 8.6 GiB at the start, 7.5 GiB at about 12:15 and 8.4 GiB after I cleared npm's package
  cache (0.96 GiB, regenerable) under your earlier instruction today to clear caches to get 8 GB free. I could not
  find what took the other 2 GiB. Thermal state was not recorded: `pmset -g therm` reports nothing
  on this chip.
- The machine was not idle: Slack, Chrome, `node`, `terraform` and system services were among the
  busiest other processes the neighbour sampler saw, recorded in each result file under
  `neighbours`.
- Code: commits c11afda (checks and predictions), f196415 (kernels, worker, runner), d394bea (the
  protocol, written before the first gated run), c073559 (report), 44565cd (mutation checks).

## What I did

1. Wrote the five predictions and the checks that judge them, with tests on synthetic data, and
   committed them before any kernel existed.
2. Wrote three Metal kernels through `mx.fast.metal_kernel`: a STREAM triad (one element per thread,
   or four through `float4`), the same triad with a threadgroup-memory array of 1 to 32 KiB declared
   and used by every thread, and a compute loop of 8 independent fused multiply-add chains in float32
   and float16. Thread 0 of each kernel writes the threadgroup size and SIMD-group width it ran with,
   so the sweep shows what actually ran. A smoke run on 64 Ki elements checked that all 30
   configurations validated and dispatched as asked, and printed no timings.
3. Fixed the protocol before the first gated run: `python scripts/day04_run.py --rest 120
   --wait-quiet 900`, a 120 s rest before every process run in every phase (Day 3 added rests only
   after seeing a failure).
4. Attempt 1, all three phases, 09:42 to 10:07: `stream` and `fma` held. `occupancy` did not (one
   size varied 11.4% against the 10% limit, one run of three slow), and its file is in
   `results/rejected/`.
5. The re-run of `occupancy` started at about 10:09 and was refused, after its 900 s quiet-wait,
   because the Mac was on battery. The session then lost its network. After you said the machine was quiet I ran
   `python scripts/day04_run.py --only occupancy --rest 120 --wait-quiet 900` at 12:19, and it held.
6. Rendered the tables from the result files (`python scripts/day04_report.py`), wrote the CUDA
   mapping, and checked each of its statements against the NVIDIA pages and MLX's installed source.

## Results

Each number is the median over 3 independent process runs, each the fastest of 8 timing samples.
Bandwidth counts the least possible traffic of the triad, 12 bytes per element (read two float32,
write one). The percentage columns compare with Day 3's MLX triad (87.2 GB/s).

#### What the kernels reported about how they ran

| Phase | Configurations | Ran at the threadgroup size asked, in every run | SIMD-group width reported |
|---|---|---|---|
| stream | 12 | 12 of 12 | 32 |
| occupancy | 6 | 6 of 6 | 32 |
| fma | 12 | 12 of 12 | 32 |

#### Streaming triad by threadgroup size

| Threadgroup size | scalar, GB/s median (min to max) | scalar, % of MLX triad | vec4, GB/s median (min to max) | vec4, % of MLX triad |
|---|---|---|---|---|
| 32 | 84.2 (83.7 to 85.3) | 97% | 83.7 (82.6 to 90.8) | 96% |
| 64 | 88.7 (86.0 to 101.3) | 102% | 83.2 (82.9 to 84.3) | 95% |
| 128 | 86.4 (84.4 to 89.0) | 99% | 84.1 (83.9 to 84.4) | 96% |
| 256 | 86.4 (84.4 to 87.0) | 99% | 82.2 (81.4 to 85.4) | 94% |
| 512 | 98.6 (90.2 to 102.9) | 113% | 84.5 (83.3 to 84.9) | 97% |
| 1024 | 87.9 (85.8 to 89.2) | 101% | 83.3 (74.0 to 83.8) | 96% |

#### Threadgroup memory and bandwidth

| Threadgroup memory per group of 256 | GB/s median (min to max) | Variation | Relative to 1024 bytes |
|---|---|---|---|
| 1 KiB | 85.5 (83.7 to 86.1) | 1.2% | 1.00 |
| 4 KiB | 84.7 (80.4 to 85.2) | 2.6% | 0.99 |
| 8 KiB | 81.4 (77.8 to 82.6) | 2.5% | 0.95 |
| 16 KiB | 93.2 (91.2 to 93.3) | 1.1% | 1.09 |
| 24 KiB | 92.6 (90.7 to 93.2) | 1.1% | 1.08 |
| 32 KiB | 80.0 (78.6 to 81.0) | 1.3% | 0.94 |

#### Fused multiply-add throughput by threadgroup size

| Threadgroup size | fp32 TFLOPS, median (min to max) | fp32 variation | fp16 TFLOPS, median (min to max) | fp16 variation |
|---|---|---|---|---|
| 32 | 2.83 (2.83 to 2.84) | 0.1% | 3.06 (3.05 to 3.06) | 0.1% |
| 64 | 2.85 (2.83 to 2.85) | 0.2% | 3.06 (3.05 to 3.06) | 0.1% |
| 128 | 2.84 (2.84 to 2.85) | 0.1% | 3.06 (3.05 to 3.06) | 0.2% |
| 256 | 2.85 (2.83 to 2.85) | 0.2% | 3.06 (3.05 to 3.06) | 0.1% |
| 512 | 2.84 (2.84 to 2.84) | 0.0% | 3.06 (3.06 to 3.06) | 0.0% |
| 1024 | 2.84 (2.84 to 2.85) | 0.1% | 3.06 (3.05 to 3.06) | 0.2% |

#### Predictions

| Prediction | Result |
|---|---|
| Best hand-written triad at least 70% of Day 3's MLX triad; threadgroup sizes 64 to 1024 within 10% of each variant's best | did not hold: scalar threadgroup 64: 90% of the variant's best, more than 10% below it; scalar threadgroup 128: 88% of the variant's best, more than 10% below it; scalar threadgroup 256: 88% of the variant's best, more than 10% below it; scalar threadgroup 1024: 89% of the variant's best, more than 10% below it |
| 32 KiB of threadgroup memory costs at least 20% of the 1 KiB bandwidth; no step up in memory more than 10% faster | did not hold: 32768 bytes of threadgroup memory cost 6% of the bandwidth at 1024 bytes, less than the predicted 20%; going from 8192 to 16384 bytes made it 14% faster, more than the 10% allowed |
| Best fp32 FMA between 2.5 and 4.5 TFLOPS and at least 0.95 times Day 3's fp32 matrix multiply; fp16 between 0.8 and 1.25 times fp32 | held |

#### Machine speed before and after each phase

| Phase | Decode probe before, tokens/s | Decode probe after, tokens/s |
|---|---|---|
| stream | 155.6 | 155.8 |
| occupancy | 156.1 | 157.3 |
| fma | 161.9 | 155.1 |

### CUDA concepts against what was measured

| CUDA concept | What the CUDA guide says (L, fetched 2026-11-24) | Nearest Metal concept | What today shows on this chip |
|---|---|---|---|
| Warp | Threads in a block are grouped 32 at a time into warps that run in SIMT fashion [1]; `warpSize` is a run-time value, commonly 32 [3] | SIMD group (`threads_per_simdgroup`) | M: every kernel in every run reported a SIMD-group width of 32 (dispatch table) |
| Thread block | Threads that can share memory and synchronize; at most 1024 threads in the compute capability 10.0 example [2] | Threadgroup | M: every threadgroup size from 32 to 1024 compiled, validated and ran as asked for all three kernels (dispatch table). Larger sizes were not tried |
| Streaming Multiprocessor (SM) | The GPU is a collection of SMs, each with registers, a unified data cache and functional units [1] | GPU core | Not measured. `system_profiler` reports 10 GPU cores for this chip |
| Shared memory | Per-block scratchpad on the SM, sharing physical storage with the L1 cache [1][2] | Threadgroup memory (`threadgroup` arrays) | M: arrays from 1 KiB to 32 KiB per threadgroup compiled and were used by every thread (occupancy table). The on-chip storage behind them is not documented here |
| `__syncthreads()` | Blocks every thread of a block until all reach it [2] | `threadgroup_barrier(mem_flags::mem_threadgroup)` | M, as correctness only: the occupancy kernel's neighbour exchange validated in every configuration, which needs the barrier |
| Occupancy | Active warps over the most an SM supports; shared memory, registers and threads per block all limit it [2] | Same idea: threadgroup memory, registers and threads per group limit how many groups a core holds | M: the occupancy table measures bandwidth against threadgroup memory declared. The occupancy itself is not read: Metal through MLX exposes no counter for it here |
| Tensor Cores, warp matrix functions | Warp-wide matrix operations of the form D = A * B + C that use Tensor Cores [3] | `simdgroup_matrix` 8 x 8 operations; MLX 0.32.4's matrix-multiply kernels (`steel/gemm/mma.h`) build on them, two elements per thread | L only: read in MLX's installed source. Whether this chip has separate matrix units is not shown by anything run today |

Sources, fetched 2026-11-24 (the old URL `cuda-c-programming-guide` now redirects to these):

1. NVIDIA, CUDA Programming Guide, 1.2 Programming model:
   https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html
2. NVIDIA, CUDA Programming Guide, 2.3 Writing CUDA kernels:
   https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html
3. NVIDIA, CUDA Programming Guide, 5.4 C++ language extensions:
   https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html

What these numbers show:

- **Every threadgroup size from 32 to 1024 ran as asked, with a SIMD-group width of 32.** That is
  30 configurations in every run. Sizes above 1024 were not tried.
- **The `float4` triad is flat across threadgroup sizes, and the scalar triad is not.** The vec4
  kernel ran at 82.2 to 84.5 GB/s at every size, each within 3% of its best and at most 97% of
  Day 3's MLX triad. The scalar kernel ran at 84.2 to 98.6 GB/s with no trend in size. Its best,
  98.6 GB/s at size 512, is 13% above the MLX triad, and it is the median of runs at 90.2, 98.6
  and 102.9. Size 64 has a run at 101.3. Those two runs are above the 100 GB/s Apple prints for
  this chip, and I do not know why (the inputs are random uniform values, and the byte count is the
  least possible). So Day 3's bandwidth roof describes MLX's kernels and is not shown to be
  this chip's ceiling.
- **More threadgroup memory did not cost bandwidth the way I predicted.** From 1 KiB to 8 KiB
  per group the bandwidth went from 85.5 to 81.4 GB/s, it then rose to 93.2 at 16 KiB and 92.6 at
  24 KiB (8% to 9% above 1 KiB), and at 32 KiB it was 80.0, 6% below 1 KiB. Every size repeated
  within 2.6%. I did not look for the reason, and the occupancy itself was not read.
- **A plain fused multiply-add loop reaches about the same rate as MLX's matrix multiply, and
  threadgroup size does not matter for it.** 2.85 TFLOPS in float32 (1.03 times Day 3's MLX fp32
  matrix multiply) and 3.06 in float16 (99% of Day 3's fp16 roof), with run-to-run variation at most 0.3% and medians within 1% across
  sizes 32 to 1024. Float16 was 1.08 times float32, again no doubling.
- **Three of the five predictions held, and two did not** (below). The first row of the
  predictions table bundles predictions 1 and 2: the first part held (the best triad was 113% of
  MLX's) and the second did not.

## What failed or surprised me

- **Prediction 3 failed on both counts.** I expected 32 KiB per group to cost at least 20% of the
  1 KiB bandwidth. It cost 6%. I expected no step up in memory to make it more than 10% faster.
  Going from 8 KiB to 16 KiB made it 14% faster. I guessed the direction and size wrongly.
- **Prediction 2 failed for the scalar kernel, narrowly.** I expected every size from 64 to 1024
  to be within 10% of the variant's best. Sizes 64, 128, 256 and 1024 were 90%, 88%, 88% and 89%
  of it, up to two points beyond the allowance. The best itself is a bimodal median, as above, so
  this says more about run-to-run spread than about size. The vec4 kernel held.
- **The occupancy phase took three tries.** Attempt 1 failed the stability limit. The second try
  was refused because the Mac was on battery. The third held. I had seen attempt 1's numbers when
  the third ran, so the occupancy result and its verdict are not blind. The predictions were still
  committed before any kernel ran.
- **The scheduled session died with its record partly wrong.** It lost its network at 10:49,
  after writing `STUCK.md` and before committing the `stream` and `fma` results, so the note said
  they were committed when they were not. I audited its five commits (signed off, no attribution
  lines, no em dashes, 388 tests passing, lint clean), checked the files against its log, committed
  the results, and recorded the correction in `LOG.md`.
- **The sampler flagged every phase as contended** (peaks of 260% to 355% of a core from the other
  processes in attempt 1, 265% in the final occupancy run). Its window includes the 120 s rests, so
  it cannot say whether anything overlapped the timed work. The run-to-run variation decides
  stability, as on Day 3.

## Not claimed

- **Why the scalar triad's runs split between about 85 and about 100 GB/s**, or why two runs were
  above Apple's printed 100 GB/s.
- **Why 16 and 24 KiB of threadgroup memory were faster than 8 KiB**, or why 32 KiB was slower.
- **Occupancy.** Metal through MLX exposes no occupancy counter here, so the occupancy table
  measures bandwidth against memory declared, not occupancy.
- **That any threadgroup size is best.** The scalar medians do not trend with size.
- **Anything about CUDA hardware.** The CUDA column is NVIDIA's guide as fetched on 2026-11-24, and
  the tensor-core row rests on NVIDIA's guide and a reading of MLX's source. Whether this chip has separate matrix
  units is not shown by anything run today.

## Limits

- One machine, one day, not an idle machine: the sampler flagged every phase contended, and GPU
  use by other processes was checked only at the start of each phase, not during the runs.
- The triad check compares the first 1 Mi of 64 Mi output elements with NumPy, so a kernel that
  skipped work in the tail would not be caught, and the bandwidth assumes all 64 Mi were done.
- One array size (256 MiB per array), threadgroup sizes up to 1024 only, and one compute-loop shape
  (8 independent chains). The occupancy sweep varies the memory declared, in 1 KiB to 32 KiB steps.
- The occupancy result is the final of three tries, as above.
- The 10-core GPU count is what `system_profiler` reports, not something measured.

## Evidence ledger entries

The rows added to [EVIDENCE.md](../../EVIDENCE.md) today:

- Hand-written kernels ran at every threadgroup size from 32 to 1024, SIMD-group width 32 (M).
- The streaming triad across threadgroup sizes, with the unexplained spread disclosed (M).
- Bandwidth against threadgroup memory declared, with the failed prediction and the rejected
  attempt disclosed (M).
- The fused multiply-add rates and their flatness across threadgroup size (M).
- The CUDA to Metal mapping, with measured and read-only parts separated (L, with M).
- Built things: the Day 4 checks, kernels, runner, report and their tests.
