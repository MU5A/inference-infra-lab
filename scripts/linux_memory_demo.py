"""What the Linux kernel does with memory when a cgroup hits its limit.

Run this INSIDE a memory-limited container, never on the host.

Four experiments, each with the cgroup's own accounting read before and after:

  page_cache         write a file, read it, drop it from the cache: how `file` memory grows
                     and shrinks
  anonymous          allocate ordinary memory until the kernel kills the process (the OOM killer)
  mmap_read          memory-map and read a file LARGER than the limit: it works, because clean file
                     pages can be evicted and read again
  mmap_private_write memory-map the same file privately and write to it: every written page becomes
                     a private copy that cannot be evicted, so this one is OOM-killed

Together they explain why a model loaded with mmap can be bigger than a container's memory limit
while the same bytes copied into ordinary memory cannot, and why weights that are modified or copied
into a GPU buffer lose that property.

Safety: the script allocates until something is killed, so it refuses to run unless its cgroup has a
memory limit of at most 1 GiB. Run it only through scripts/day02_linux.py.

Output: one JSON line per experiment, printed the moment it finishes, so that if this process is
itself killed the results already produced are not lost and the missing one shows where it died.
"""

from __future__ import annotations

import json
import mmap
import os
import subprocess
import sys
import time
from pathlib import Path

from inferlab import cgroup

MIB = 1 << 20
DATA = Path("/var/tmp/day02-data.bin")
PAGE = 4096

CHILD_ANONYMOUS = """
import os
chunks = []
total = 0
while True:
    chunks.append(bytearray(os.urandom(8 << 20)))
    total += 8
    print(total, flush=True)
"""

CHILD_PRIVATE_WRITE = """
import mmap, sys
f = open(sys.argv[1], "r+b")
m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
for offset in range(0, len(m), 4096):
    m[offset] = (m[offset] + 1) % 256
    if offset % (8 << 20) == 0:
        print(offset >> 20, flush=True)
print("done", flush=True)
"""


def emit(name: str, result: dict) -> None:
    print(json.dumps({"experiment": name, "result": result}), flush=True)
    print(f"finished {name}", file=sys.stderr, flush=True)


def mib(value: int | None) -> float | None:
    return None if value is None else round(value / MIB, 2)


def write_data(size_mib: int) -> None:
    """Write incompressible data, forcing each chunk to disk before the next. Dirty pages cannot be
    reclaimed until they are written back, so without this a fast writer can exhaust a small
    cgroup with nothing but unwritten file data and be OOM-killed itself."""
    with DATA.open("wb") as handle:
        for _ in range(size_mib // 4):
            handle.write(os.urandom(4 * MIB))
            handle.flush()
            os.fsync(handle.fileno())


def drop_from_cache(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # only clean pages can be dropped
    os.close(fd)


def read_all(path: Path) -> float:
    start = time.monotonic()
    with path.open("rb") as handle:
        while handle.read(4 * MIB):
            pass
    return time.monotonic() - start


def state() -> dict:
    snap = cgroup.snapshot()
    return {
        "current_mib": mib(snap["current"]),
        "file_mib": mib(snap["stat"].get("file")),
        "anon_mib": mib(snap["stat"].get("anon")),
    }


def run_child(code: str, *args: str) -> dict:
    before = cgroup.snapshot()["events"]
    proc = subprocess.Popen(
        [sys.executable, "-c", code, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines = proc.stdout.read().split()
    proc.wait()
    after = cgroup.snapshot()["events"]
    progress = [int(x) for x in lines if x.isdigit()]
    return {
        "returncode": proc.returncode,
        "killed_by_signal": -proc.returncode if proc.returncode < 0 else None,
        "completed": "done" in lines,
        "last_progress_mib": progress[-1] if progress else None,
        "oom_kill_events": after.get("oom_kill", 0) - before.get("oom_kill", 0),
        "limit_hit_events": after.get("max", 0) - before.get("max", 0),
    }


def page_cache(limit: int) -> dict:
    size_mib = max(4, (limit // MIB // 2) // 4 * 4)
    stages = {"start": state()}
    write_data(size_mib)
    stages["after_write"] = state()
    warm = read_all(DATA)
    stages["after_read"] = state()
    drop_from_cache(DATA)
    stages["after_drop"] = state()
    cold = read_all(DATA)
    stages["after_cold_read"] = state()
    DATA.unlink()
    return {
        "file_mib": size_mib,
        "stages": stages,
        "read_seconds": {"from_cache": round(warm, 4), "after_drop": round(cold, 4)},
    }


def mmap_read(limit: int) -> dict:
    size_mib = (2 * limit // MIB) // 4 * 4
    write_data(size_mib)
    drop_from_cache(DATA)
    before = cgroup.snapshot()
    peak_current = peak_file = 0
    touched = 0
    with DATA.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        for offset in range(0, len(mapped), PAGE):
            _ = mapped[offset]
            touched += PAGE
            if touched % (8 * MIB) == 0:
                snap = cgroup.snapshot()
                peak_current = max(peak_current, snap["current"] or 0)
                peak_file = max(peak_file, snap["stat"].get("file", 0))
        mapped.close()
    after = cgroup.snapshot()
    return {
        "file_mib": size_mib,
        "limit_mib": mib(limit),
        "completed": True,
        "bytes_touched_mib": mib(touched),
        "peak_current_mib": mib(peak_current),
        "peak_file_mib": mib(peak_file),
        "limit_hit_events": after["events"].get("max", 0) - before["events"].get("max", 0),
        "oom_kill_events": after["events"].get("oom_kill", 0) - before["events"].get("oom_kill", 0),
    }


def main() -> int:
    try:
        limit = cgroup.require_limit()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    info = {
        "kernel": os.uname().release,
        "limit_mib": mib(limit),
        "swap_limit_mib": mib(cgroup.snapshot()["swap_limit"]),
        "page_size": os.sysconf("SC_PAGE_SIZE"),
        "cgroup_v2": (cgroup.DEFAULT_ROOT / "cgroup.controllers").exists(),
    }
    emit("info", info)
    emit("page_cache", page_cache(limit))
    emit("anonymous", run_child(CHILD_ANONYMOUS))
    emit("mmap_read", mmap_read(limit))
    emit("mmap_private_write", run_child(CHILD_PRIVATE_WRITE, str(DATA)))
    DATA.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
