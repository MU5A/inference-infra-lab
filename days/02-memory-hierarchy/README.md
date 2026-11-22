# Day 2: memory hierarchy and Linux internals

**Stage:** 1, systems foundations. **Label:** M (measured on this machine). **Status:** in progress.
The Linux container experiments are done and reported below. **The C microbenchmarks have not been
measured yet.**

## Goal

Find, by measurement, where this chip's memory hierarchy changes behaviour: how fast memory can be
streamed, how long a dependent load takes at each level, how much of a matrix multiply is lost to
memory access, what a cold start costs, and what Linux does with memory when a limit is reached.
Day 3 builds the roofline model from these numbers, and Day 6 uses them to predict decode speed.

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
- **The first test of the plateau finder failed 12 of 21 tests** because my synthetic latency curve
  was too sparse to contain a narrow cache level. The real sweeps use four points per octave.
- **Docker Desktop would not start** until I stopped an orphaned background process left over from
  an earlier session. Its build cache, 4.4 GB, was pruned with your approval and free disk rose from
  5.5 to 8.8 GiB.
- **The machine has 2.3 GB of swap in use**, so the pre-flight now also checks free memory, as the
  system itself reports it, before any benchmark.

## What is not done

- **The C microbenchmarks have not been measured.** Memory bandwidth, load latency by working-set
  size, matrix-multiply throughput and cold file reads are built, tested and validated, but the run
  (`scripts/day02_run.py`) needs a quiet machine, and the pre-flight refused it earlier (GPU 29% busy
  from a video player, load average 5.3). No Day 2 number below this line is claimed.
- **Day 1's three open measurements** are repeated after Day 2's, under the same checks.

## What the OS reports about this chip

For comparison with the measured plateaus, read before measuring:

| | Performance cores (4) | Efficiency cores (4) |
|---|---|---|
| L1 data cache | 128 KB | 64 KB |
| L2 cache | 16 MB (shared) | 4 MB (shared) |

Cache line 128 bytes, page size 16 KB, no L3 reported.

## Limits

- The containers run inside Docker Desktop's Linux virtual machine (kernel 7.0.12-linuxkit, 4 KiB
  pages), not on macOS. This is how Linux behaves under a limit. It is not how macOS behaves.
- A 96 MiB limit is tiny on purpose, so the experiments finish in seconds. The behaviour is the same
  at larger sizes, but the numbers are specific to this limit.
- The read times in the page cache table are not SSD speeds. The virtual disk is a file on the host,
  which macOS may itself have cached.
- Three runs of each. All three agreed on every outcome, so there is no spread to report.
