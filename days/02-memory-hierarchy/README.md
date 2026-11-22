# Day 2: memory hierarchy and Linux internals

**Stage:** 1, systems foundations. **Label:** M (measured on this machine). **Status:** in progress,
**no measurements taken yet.**

## Goal

Find, by measurement, where this chip's memory hierarchy changes behaviour: how fast memory can be
streamed, how long a dependent load takes at each level, how much of a matrix multiply is lost to
memory access, and what a cold start costs when a model file is not yet in memory. Day 3 builds the
roofline model from these numbers, and Day 6 uses them to predict decode speed.

## What exists

The microbenchmarks, the analysis that reads cache levels off a latency curve, and the runner are
written and tested. See [LOG.md](LOG.md) for what was built and the mistakes the tests caught.

## What is not done

- The measurement run. It was refused by the pre-flight check (GPU 29% busy, load average 5.3) and
  by the disk requirement. Results will appear here, generated from the results file, when it runs.
- The Linux container experiments: page cache, cgroup memory limits, the OOM killer, and file-backed
  versus anonymous memory under a limit.

## What the OS reports about this chip

For comparison with the measured plateaus, read before measuring:

| | Performance cores (4) | Efficiency cores (4) |
|---|---|---|
| L1 data cache | 128 KB | 64 KB |
| L2 cache | 16 MB (shared) | 4 MB (shared) |

Cache line 128 bytes, page size 16 KB, no L3 reported.
