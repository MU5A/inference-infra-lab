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

## 6. The Linux container experiments

Docker Desktop was started for these, after the quiet-window benchmarks were refused. It would not
launch until an orphaned backend process from an earlier session was force-stopped (checked first:
its parent was pid 1 and its VM was long gone). With approval, Docker's unused build cache (4.4 GB)
was pruned, and free disk rose from 5.5 to 8.8 GiB. The image `python:3.12-alpine` (18.5 MB
compressed, arm64, official) was pulled with approval and is pinned by digest in the results.

1. **First run: exit 137, no output.** The container's main process was killed. The script printed
   only at the end, so nothing survived. `memory.oom.group` was 0, so the kernel killed processes
   individually.
2. **Fix and a second hypothesis.** My leading suspect was my own writer, which called `fsync` once at
   the end, so I synced every chunk and the script ran to the end. That is a coincidence until the
   failure is reproduced on purpose.
3. **A direct test killed the container again.** Adding the unsynced write as a fifth experiment gave
   exit 137 twice and no result for it, but the parent died with it, so it was not a clean test.
4. **A clean A/B.** One container whose only process writes the same 192 MiB, with and without a sync
   after each chunk, using Docker's own `OOMKilled` flag. Prediction recorded before the run:
   unsynced killed, synced completes. Result: three pairs out of three, as predicted.
5. **Results** are in `results/linux_memory.json` and the README. Every prediction held in all
   three runs.

## 7. A memory check added to the pre-flight

`memory_pressure` reported 56% free but `vm.swapusage` showed 2.3 GB of 3 GB in use, so this
machine has been under pressure and a benchmark with a 1 GiB working set could be disturbed. The
pre-flight now refuses below 30% free memory, and every results file records swap use. Swap in use
alone is not refused, since old pages can sit in swap without causing contention now. Tested, with
two mutations.

## 8. The measurement run, and four attempts at one group

The C microbenchmarks were run through `scripts/day02_run.py` after you said the machine was quiet.
The pre-flight was not clean at that moment (the GPU was 8% busy on average, with VLC still running),
so the runner waited, then proceeded when it passed. Free disk was 5.0 to 5.6 GiB, below the 8 GB
rule, and I ran on your instruction.

- **First run.** Bandwidth was stable. Latency on the performance cores was stable, with the same
  four plateaus in all three sweeps. The efficiency-core latency sweep was not (6, 4 and 7 plateaus).
  Compute was not stable: 19 of 24 entries failed, with every variant slower by about the same factor
  in each later run at n=128 to 512. The cold read was stable but had no control (see below).
  I looked at what was running: another Claude session was compiling and running tests.
- **A sampler for neighbours.** A pre-flight at the start cannot see a build that starts mid-run.
  `Watch` now records the busiest other processes every two seconds during each group, with names
  only. It attributes interference. It does not decide stability, which is still the run-to-run
  variation.
- **Attempt 2 of compute** (also the redesigned cold-read, run together): 19 of 24 compute entries
  were stable and the 5 unstable ones were all n=128. At 128 x 128 a call takes microseconds, so
  timing one call measured the timer. A flaw in my benchmark, not in the machine.
- **Attempt 3**: each timed sample now repeats the call until it lasts at least 20 ms. Seven entries
  were unstable and the sampler showed `compile` at 96% and `go` at 38% CPU. Interference can only
  slow a sample down, so a single bad sample in a best-of-3 spoils the run.
- **Attempt 4**: best of 8 samples per run. Stable on all 24 entries, with other processes peaking at
  47% CPU. I changed the benchmark after each failure, so all four attempts stay in the table.

The cold-read group was redesigned after the first run showed a memory-mapped touch at 0.07 GB/s,
about 100 times slower than the cached file. That could be the disk or the access pattern, and there
was no cold sequential read to tell them apart. The redesign gives each way (touch, sequential read,
WILLNEED, SEQUENTIAL) its own freshly made cold copy and has predictions written before the run.
One prediction failed (see the README).

## 9. The full mutation check

Run once on the final code, 42 deliberate breakages. 41 were caught. One was missed: the
free-memory regex could read any percentage, because my test input held only one. A distractor
percentage was added to the test and the breakage is now caught, so all 42 fail the suite. This is
the second time a full run found a gap that the piecewise runs had not.

## 10. What is left

The measurement run (`scripts/day02_run.py`) has not been taken. The pre-flight refused it: the GPU
was 29% busy and the 1-minute load average was 5.3. Free disk was also 5.7 GiB, below the 8 GB the
runbook requires. Nothing in `README.md` or the evidence ledger claims a Day 2 result.
Nothing in Day 2 is outstanding except what the README lists under "Not claimed". Day 1's three
open measurements are still to be done.
