# Day 2 log

A running record of what was done, in order, including the mistakes. No timings are claimed here:
the measurement run has not happened yet (see the end).

## 0. Closing out Day 1 first

Day 1 still had three measurements open (harness overhead, cold versus warm prompt cache,
llama-bench). They were attempted at the start of the day. What happened is recorded in the Day 1
log, section 10, because it changed the harness: a second llama-bench run decoded at 60 tokens/s
with a standard deviation of 50 while a video player was using the GPU, and the pre-flight passed
because it only looked at CPU load.

## 1. Two safeguards added to the harness before any Day 2 measurement

- The pre-flight now samples GPU utilization (`ioreg`, no privileges needed) and refuses at a peak
  above 25% or a mean above 5%. Video playback showed 51% in one sample and 0% in the next, which is
  why it samples about 2 seconds, not once.
- Results now carry a coefficient-of-variation check. Decode must repeat within 5% and time to first
  token within 10%. The trusted Day 1 baseline had 1.8 to 3.3% and under 1.7%. The disturbed runs
  had 5 to 7% and, worst, 83%. Unstable results are written and marked, never dropped.

Each safeguard has tests and mutations (a busy GPU never refused, the limit disabled, the flag
forced true, and so on), all caught.

## 2. What the OS says about this chip, read before measuring

`sysctl` reports performance cores with a 128 KB L1D and a 16 MB shared L2, efficiency cores with a
64 KB L1D and a 4 MB L2, a 128-byte cache line and 16 KB pages. No L3 is reported. These are the
numbers the measured latency plateaus will be compared with, so they are recorded now, not after.

## 3. The microbenchmarks (C, in `microbench/`)

| Program | What it measures |
|---|---|
| `membw` | Sustained bandwidth, STREAM-style (copy, scale, add, triad), 1 to 8 threads, arrays of 256 MiB each so they cannot sit in any cache |
| `chase` | Load-to-use latency by pointer chasing over a random single cycle, 4 KiB to 1 GiB, 4 points per octave |
| `matmul` | Single-precision matrix multiply: naive, loop-reordered, tiled, and Apple's Accelerate library |
| `mmap_touch` | Time to fault in a memory-mapped file, one touch per page, the way llama.cpp loads weights |
| `copy_nocache` | Copy a file without leaving it in the page cache, to create a cold file |

Each validates its own result: STREAM arrays are checked against a scalar replay, the pointer
chain must be one full cycle, every matmul variant must match a double-precision reference, and
the memory-mapped checksum is checked independently in Python.

## 4. Mistakes caught by the tests before any measurement

1. I left a nonsense placeholder in the tail of `chase.c` while composing it. It would not have
   compiled.
2. `chase` segfaulted on its first self-test. macOS's `aligned_alloc` returns NULL for sizes that
   are not a multiple of the alignment, and I did not check for NULL. Nearly every size in the real
   sweep would have failed. Fixed with `posix_memalign` and a NULL check.
3. The plateau finder failed 12 of 21 tests. The code was right and my synthetic curve was too
   sparse: at 2 points per octave a level two octaves wide has only 2 points, below the 3-point
   minimum for calling something a plateau. This is a real sensitivity of the method, so the real
   sweeps use 4 points per octave and a test records the failure mode.
4. `matmul` validated only its warm-up result, not the timed repetitions. A variant that is right
   on its first call and wrong afterwards would have reported a fast, wrong number. Found while
   thinking about which mutations to write, fixed, and now caught by a mutation.

## 5. Mutation check

Eleven deliberate breakages of the C code and the analysis module, all caught (Fisher-Yates instead
of Sattolo, a pointer chain where every node points at itself, wrong triad, a scale kernel that does
nothing, a matmul that does not zero its output, tiles running past the edge, an off-by-one stride,
a copy that appends, and three analysis-limit changes).

## 6. Not done yet

The measurement run (`scripts/day02_run.py`) has not been taken. The pre-flight refused it: the GPU
was 29% busy and the 1-minute load average was 5.3. Free disk was also 5.7 GiB, below the 8 GB the
runbook requires. Nothing in `README.md` or the evidence ledger claims a Day 2 result.
The Linux container part (page cache, cgroup memory limits, the OOM killer) has not been started: it
needs Docker, which is stopped, and an image download that has not been approved.
