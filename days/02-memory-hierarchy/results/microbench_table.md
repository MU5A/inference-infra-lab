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
