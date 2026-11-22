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
