"""Write a file of N MiB in 4 MiB chunks, optionally forcing each chunk to disk.

Run inside a memory-limited container by scripts/day02_linux.py, as the container's only process.
usage: dirty_write.py SIZE_MIB sync|nosync
"""

import os
import sys

size_mib, mode = int(sys.argv[1]), sys.argv[2]
with open("/var/tmp/dirty.bin", "wb") as handle:
    for _ in range(size_mib // 4):
        handle.write(os.urandom(4 << 20))
        handle.flush()
        if mode == "sync":
            os.fsync(handle.fileno())
print("done")
