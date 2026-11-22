# Day 2: memory hierarchy and Linux internals

**Stage:** 1, systems foundations. **Label:** M (measured on this machine). **Status:** done, with the
exceptions listed under "Not claimed". Day 1's three open measurements are still outstanding.

## Goal

Find, by measurement, where this chip's memory hierarchy changes behaviour: how fast memory can be
streamed, how long a dependent load takes at each level, how much matrix-multiply throughput depends
on how the loops are written, what a cold start costs, and what Linux does with memory when a limit
is reached. Day 3 builds the roofline model from these numbers, and Day 6 uses them to predict
decode speed.

How the results were judged: each group ran three independent times and had to repeat within limits
fixed before the first run (10% variation for bandwidth, latency and compute, 25% for cold reads from
disk). A group that did not repeat was marked unstable, kept on record, and not cited. All
files, and which may be cited, are listed in [`results/README.md`](results/README.md).

## Memory and compute on this chip (measured)

Sustained memory bandwidth, STREAM-style, 256 MiB arrays, 3 independent runs:

| Threads | Triad GB/s, median (min to max) | Copy GB/s, median | Variation (CV) of triad |
|---|---|---|---|
| 1 | 82.2 (81.6 to 84.2) | 80.2 | 1.3% |
| 2 | 83.5 (82.8 to 84.7) | 85.8 | 0.9% |
| 3 | 84.1 (82.6 to 84.3) | 86.5 | 0.9% |
| 4 | 85.1 (84.4 to 86.1) | 84.8 | 0.9% |
| 6 | 81.3 (80.5 to 83.5) | 82.3 | 1.6% |
| 8 | 81.6 (80.3 to 83.2) | 82.8 | 1.4% |

Load-to-use latency by pointer chasing, performance cores, 3 independent sweeps of 4 points per octave:

| Working set | Latency, ns (min to max over 3 runs) |
|---|---|
| 4 KiB | 1.1 (1.1 to 1.1) |
| 64 KiB | 1.1 (1.1 to 1.1) |
| 128 KiB | 1.1 (1.1 to 1.1) |
| 256 KiB | 4.6 (4.4 to 4.7) |
| 1.0 MiB | 6.2 (5.6 to 6.3) |
| 4.0 MiB | 6.3 (6.0 to 7.1) |
| 8.0 MiB | 7.8 (7.4 to 8.5) |
| 11.3 MiB | 7.9 (7.1 to 15.3) |
| 16.0 MiB | 23.3 (16.0 to 47.7) |
| 22.6 MiB | 62.7 (45.0 to 66.7) |
| 32.0 MiB | 88.9 (80.9 to 100.3) |
| 64.0 MiB | 99.0 (98.0 to 101.3) |
| 256.0 MiB | 104.8 (104.2 to 105.4) |
| 1024.0 MiB | 106.6 (104.3 to 109.5) |

The plateau finder found [4, 4, 4] plateaus in the three sweeps. The first (median 1.11 ns, variation 1.9%) ends at 128 KiB in every run. The last (median 103.4 ns, variation 0.6%) is main memory. The ends of the two middle plateaus were [4.0, 0.8, 2.0] and [11.3, 6.7, 11.3] MiB in the three runs, so they do not locate a capacity.

Matrix multiply, median over 3 process runs of the best of 8 samples in each (largest variation of any entry: 1.5%):

| GFLOPS, single precision | n=128 | n=256 | n=512 | n=1024 |
|---|---|---|---|---|
| naive loop (i, j, k) | 3.6 | 2.8 | 2.4 | 2.2 |
| reordered loop (i, k, j) | 38.4 | 29.5 | 29.9 | 29.7 |
| tiled, 32 x 32 | 18.9 | 18.9 | 18.8 | 17.2 |
| tiled, 64 x 64 | 27.9 | 28.0 | 25.7 | 21.3 |
| Accelerate library, default threads | 1303.4 | 1530.9 | 1653.6 | 1563.3 |
| Accelerate with VECLIB_MAXIMUM_THREADS=1 | 1300.4 | 1511.0 | 1657.3 | 1560.7 |

Every attempt at the matrix-multiply group, kept on record:

| Attempt | What changed | Result |
|---|---|---|
| 1 | First run, no neighbour sampling | 19 of 24 entries unstable; at n=128 to 512 every variant was slower by about the same factor in each later run |
| 2 | Neighbour sampling added | 5 of 24 entries unstable; all at n=128, where one call takes microseconds |
| 3 | Each timed sample repeated to at least 20 ms | 7 of 24 entries unstable; another process peaked at 155% CPU |
| 4 | Best of 8 samples per run instead of 3 | 0 of 24 entries unstable (accepted) |

Cold reads of the 469 MB model file, each pass on its own fresh copy made with the page cache bypassed, 3 trials:

| First pass over a cold file | GB/s, median (min to max) | Variation (CV) | Stable |
|---|---|---|---|
| memory-mapped, one touch per page | 0.19 (0.10 to 0.22) | 29.7% | no |
| sequential read() | 2.43 (2.32 to 2.48) | 2.8% | yes |
| memory-mapped, madvise(WILLNEED) | 0.97 (0.41 to 1.11) | 36.6% | no |
| memory-mapped, madvise(SEQUENTIAL) | 0.19 (0.15 to 0.21) | 12.8% | yes |

A cold sequential read was 10.7 to 24.8 times faster than a cold memory-mapped touch in every trial. Predictions written before the run that failed: ['trial 3: a willneed hint did not make a cold touch 3x faster'].

What these numbers show:

- **Streaming bandwidth is about 80 to 86 GB/s, and one thread already gets there.** The triad
  kernel ran between 80.3 and 86.1 GB/s for every thread count from 1 to 8, so adding threads added
  almost nothing: one thread already reached the rate that eight did.
- **Level 1 is 128 KiB, and that is measured, not read from a datasheet.** Latency is 1.1 ns up to
  exactly 128 KiB and jumps just above it. The operating system reports 128 KB for the performance
  cores' L1 data cache, so the two agree.
- **Level 2 is a gentle slope, not a clean step.** From 256 KiB to about 11 MiB latency rises
  slowly from 4.6 to about 8 ns. The OS reports a 16 MiB L2 and the curve reaches 23 ns at 16 MiB
  and about 89 ns at 32 MiB, so the cliff sits around the reported size. I did not separate cache
  effects from page-table effects (pages are 16 KiB), and the plateau finder's middle boundaries did
  not repeat between runs, so I make no claim about where L2 ends beyond that.
- **Main memory latency is about 103 ns** (variation 0.6% between sweeps). It is already 99 ns at
  64 MiB.
  Between 16 and 64 MiB there is a slope, not a second plateau, so I claim no separate cache level
  there.
- **Loop order is worth about 10 to 14 times; tiling did not help here.** The reordered loop is
  about 30 GFLOPS against 2.2 to 3.6 for the textbook loop. My tiled versions were slower than the
  simple reordered loop. At n=1024 the three matrices total 12 MiB, which fits in the 16 MiB L2, so
  there is little for tiling to save, and my tiling adds loop overhead.
- **The Accelerate library is about 700 times faster than the textbook loop** (1.3 to 1.65 TFLOPS).
  Setting `VECLIB_MAXIMUM_THREADS=1` changed nothing. Either the matrix hardware does the work
  regardless of threads, or the setting was not honoured. I did not tell these apart.
- **A cold sequential read of the model file ran at about 2.4 GB/s. Reading the same file by
  touching memory-mapped pages ran 11 to 25 times slower** (0.10 to 0.22 GB/s, unstable). The data
  is the same and the disk is the same, so the difference lies in how the pages arrive. That is
  consistent with one small read per page fault against one large streaming read, but I did not
  trace the I/O. `madvise(SEQUENTIAL)` made no difference. `madvise(WILLNEED)` helped in every
  trial, by 11, 4.5 and 2.1 times over a plain touch.

## Linux: what the kernel does when a memory limit is reached (measured)

A model file is loaded into memory one of two ways, and a container's memory limit treats them very
differently. These experiments run in fresh containers with a 96 MiB limit and no swap, three times
each, and show the difference with the cgroup's own accounting. The predictions were written down in
`src/inferlab/cgroup.py` before the first run.

Image `python@sha256:c5ba3deef00c...`, Docker 29.8.0 kernel 7.0.12-linuxkit, memory limit 96m with no swap, 3 fresh containers. Every prediction written before the first run held: True.

| Experiment | Outcome | Detail (range over the runs) |
|---|---|---|
| Ordinary memory allocated until the limit | OOM-killed (3 of 3) | killed after 72 MiB allocated; the limit was reached 37 to 39 times |
| Read a memory-mapped 192 MiB file, twice the limit | completed, no kill (3 of 3) | total memory peaked at 95.89 to 95.97 MiB against a limit of 96; the limit was reached 117 to 125 times |
| Write to a privately mapped file | OOM-killed (3 of 3) | killed after 80 MiB written |
| Write a 192 MiB file without fsync | OOM-killed (3 of 3) | Docker reports OOMKilled, exit code 137 |
| The same write with fsync after each chunk | completed (3 of 3) | exit code 0 |

Page cache accounting for a 48 MiB file (memory the cgroup attributes to cached file data):

| Stage | File memory, MiB |
|---|---|
| Before | 0.02 |
| After writing the file | 48.02 |
| After reading it | 48.02 |
| After asking the kernel to drop it (fadvise DONTNEED) | 0.02 |
| After reading it again | 48.02 |

Reading the file took 0.0068 to 0.0072 s from the cache and 0.0139 to 0.0214 s after it was dropped.

What this shows:

- **A memory-mapped file can be larger than the memory limit.** A 192 MiB file was read completely
  inside a 96 MiB limit. The kernel evicted clean file pages and read them back, so total memory sat
  at the limit while the limit was hit over a hundred times, and nothing was killed.
- **The same bytes in ordinary memory cannot.** Allocating ordinary memory was OOM-killed after 72 MiB
  in every run.
- **Mapped memory loses that property when it is written to.** Writing to a privately mapped file
  turns each written page into a private copy that cannot be evicted, and the process was killed.
  For a model this is the difference between weights that are only read and weights that are
  modified or copied into another buffer.
- **Unwritten file data can kill a process on its own.** The identical 192 MiB write, with only
  `fsync` after each chunk changed, completed in every run, and was OOM-killed in every run without
  it.

## What failed or surprised me

- **Compute took four attempts, and the failures were instructive.** Attempt 1 was disturbed by
  another process: every variant slowed by about the same factor from run to run. I added a sampler
  that records which other processes use the CPU during each run. In attempt 3 it showed compiler
  and `go` processes at up to 96% and 38% CPU, which fits the other Claude session's Go builds seen
  earlier, though it records names, not sessions. Attempt 2 exposed a
  flaw in my own benchmark: at n=128 a multiply takes microseconds, so timing one call measured the
  clock. Attempt 3 fixed that by repeating calls until a sample lasts 20 ms. Attempt 4 took the best
  of 8 samples instead of 3, because interference can only slow a sample, never speed it up. I
  changed the benchmark after seeing each failure, so the number of attempts is part of the
  evidence, and every attempt is in the table above.
- **The load average was clean when each run started and told me nothing about the next two
  minutes.** It also climbed after my own heavy runs, which then blocked the next step.
- **My first container script died with exit code 137 and no output.** It wrote its test file with
  one `fsync` at the end. I changed it to sync each chunk and it stopped dying, which proves nothing
  by itself, so I added a clean A/B: one container whose only process writes the same data, with and
  without the sync. Docker's own `OOMKilled` flag settled it, three pairs out of three. This is
  consistent with dirty pages that cannot be reclaimed until they are written back. I did not
  record the cgroup's dirty-page counter, so the mechanism is the documented explanation, not
  something I observed directly.
- **An experiment that kills processes can kill the one reporting on it.** In one variant the whole
  container died, so the script now prints each result the moment it finishes and a missing result
  shows where the kill happened.
- **The plateau finder failed 12 of its first 21 tests** because my synthetic latency curve was too
  sparse to contain a narrow cache level. The real sweeps use four points per octave.
- **My first cold-read experiment had no control.** It showed a memory-mapped touch at 0.07 GB/s, but
  without a cold sequential read I could not say whether the disk or the access pattern was slow.
  The redesigned experiment added the control, with predictions written first. One prediction
  failed: `WILLNEED` did not make a cold touch three times faster in all trials.
- **Docker Desktop would not start** until I stopped an orphaned background process left over from
  an earlier session. Its build cache, 4.4 GB, was pruned with your approval and free disk rose from
  5.5 to 8.8 GiB.
- **The machine has 2.3 GB of swap in use**, so the pre-flight now also checks free memory, as the
  system itself reports it, before any benchmark.
- **Disk was 5.0 to 5.6 GiB during the final runs**, below the 8 GB the runbook requires. I ran with
  your instruction to proceed, because this day needs at most one 469 MB copy at a time, and the
  exception is recorded here.

## Not claimed

- **The efficiency cores.** I asked macOS to run a sweep at background priority to reach them. The
  three sweeps found 6, 4 and 7 plateaus, so they disagree, and macOS offers no way to pin a thread
  to a core. The bandwidth runs at background priority gave the same speeds as the others, which
  shows only that I cannot tell which cores they used.
- **Exact memory-mapped cold-read speeds.** Only the ratio to a sequential read and the range are
  given, because those runs did not repeat within the limit.
- **Why Accelerate ignores the thread limit, if it does.**
- **A comparison with Apple's published bandwidth.** I did not look it up, so Day 3's roofline will
  fetch and cite it with a date.
- **Day 1's three open measurements.** They are still not done.

## What the OS reports about this chip

For comparison with the measured plateaus, read before measuring:

| | Performance cores (4) | Efficiency cores (4) |
|---|---|---|
| L1 data cache | 128 KB | 64 KB |
| L2 cache | 16 MB (shared) | 4 MB (shared) |

Cache line 128 bytes, page size 16 KB, no L3 reported.

## Limits

- One machine, and not an idle one: other processes were using up to about 47% of a core during the
  accepted compute run, and the bandwidth and latency runs predate the sampler that records this.
  They are accepted on the strength of their repeatability, not on a proof that nothing else ran.
- The containers run inside Docker Desktop's Linux virtual machine (kernel 7.0.12-linuxkit, 4 KiB
  pages), not on macOS. This is how Linux behaves under a limit. It is not how macOS behaves.
- A 96 MiB limit is tiny on purpose, so the experiments finish in seconds. The behaviour is the same
  at larger sizes, but the numbers are specific to this limit.
- The matrix-multiply figures are single precision and single process. Accelerate's result says
  nothing about other libraries, and none of this is a model of an inference engine.
- The read times in the Linux page-cache table are not SSD speeds. The virtual disk is a file on the
  host, which macOS may itself have cached.
