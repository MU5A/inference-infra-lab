"""Read a Linux cgroup v2's memory accounting.

Thin file reads over a root directory (so tests can point it at a fake one) plus pure parsing.
Used by the Day 2 container experiments, which show what the kernel does with page cache, anonymous
memory and memory-mapped files when a cgroup runs out of memory.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_ROOT = Path("/sys/fs/cgroup")
GIB = 1 << 30

# Size of the file written in the unsynced-versus-synced comparison, twice the 96 MiB limit. The
# container runner and the report both use this, so the figure cannot drift between them.
DIRTY_WRITE_MIB = 192


def parse_flat_keyed(text: str) -> dict[str, int]:
    """Parse the 'name value' lines of memory.stat and memory.events. Other lines are ignored."""
    parsed: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            parsed[parts[0]] = int(parts[1])
    return parsed


def _read(root: Path, name: str) -> str | None:
    try:
        return (root / name).read_text().strip()
    except OSError:
        return None


def _int(text: str | None) -> int | None:
    return int(text) if text and text.lstrip("-").isdigit() else None


def limit_bytes(root: Path = DEFAULT_ROOT) -> int | None:
    """The memory limit in bytes, or None when there is none ('max') or it cannot be read."""
    return _int(_read(root, "memory.max"))


def snapshot(root: Path = DEFAULT_ROOT) -> dict:
    return {
        "current": _int(_read(root, "memory.current")),
        "peak": _int(_read(root, "memory.peak")),
        "limit": limit_bytes(root),
        "swap_limit": _int(_read(root, "memory.swap.max")),
        "stat": parse_flat_keyed(_read(root, "memory.stat") or ""),
        "events": parse_flat_keyed(_read(root, "memory.events") or ""),
    }


def require_limit(root: Path = DEFAULT_ROOT, max_bytes: int = GIB) -> int:
    """Refuse to continue unless this cgroup has a memory limit of at most `max_bytes`.

    The experiments allocate memory until the kernel kills something. Without a small limit that
    would be the whole machine, so they must not run outside a limited container.
    """
    limit = limit_bytes(root)
    if limit is None:
        raise RuntimeError("refusing to run: this cgroup has no memory limit")
    if limit > max_bytes:
        raise RuntimeError(
            f"refusing to run: the memory limit is {limit} bytes, above the {max_bytes} allowed"
        )
    return limit


def check_dirty_write(unsynced: dict, synced: dict) -> list[str]:
    """Prediction, written after the first version of the container script died and before this
    test was run: a file twice the cgroup limit written WITHOUT syncing it as it goes exhausts the
    cgroup with dirty pages that cannot be reclaimed, and the writer is OOM-killed. The same write
    WITH a sync after each chunk completes. Only the sync differs between the two."""
    failures = []
    if not unsynced["oom_killed"]:
        failures.append("dirty_write: the unsynced write was not OOM-killed")
    if synced["oom_killed"] or synced["exit_code"] != 0:
        failures.append("dirty_write: the synced write did not complete")
    return failures


def check_run(out: dict) -> list[str]:
    """Which of the predictions made BEFORE the first run did this run's results violate?

    The predictions, from how Linux memory accounting is documented to work:
      page_cache          a written file stays in cache; dropping it empties the cache; reading it
                          again refills it
      anonymous           ordinary memory allocated past the limit is OOM-killed (SIGKILL)
      mmap_read           a file larger than the limit can be mapped and fully read with no kill,
                          because clean file pages are evicted and re-read; the limit is hit
      mmap_private_write  writing to a privately mapped file turns pages into private copies that
                          cannot be evicted, so this is OOM-killed
    An empty list means every prediction held. A non-empty one is a finding, not an error.
    """
    failures: list[str] = []
    missing = [
        e
        for e in ("info", "page_cache", "anonymous", "mmap_read", "mmap_private_write")
        if e not in out
    ]
    if missing:
        # A missing result means the process was killed before it finished that experiment.
        return [f"{name}: no result, the process died before finishing it" for name in missing]
    limit = out["info"]["limit_mib"]

    pc = out["page_cache"]
    size = pc["file_mib"]
    stages = pc["stages"]
    if stages["after_write"]["file_mib"] < 0.8 * size:
        failures.append("page_cache: the written file did not stay in the page cache")
    if stages["after_drop"]["file_mib"] > 0.25 * size:
        failures.append("page_cache: dropping the file did not empty the page cache")
    if stages["after_cold_read"]["file_mib"] < 0.8 * size:
        failures.append("page_cache: reading the file did not refill the page cache")

    anon = out["anonymous"]
    if anon["killed_by_signal"] != 9 or anon["oom_kill_events"] < 1:
        failures.append("anonymous: allocation past the limit was not OOM-killed")
    elif anon["last_progress_mib"] is None or anon["last_progress_mib"] >= limit:
        failures.append("anonymous: the process claims to have allocated more than the limit")

    mm = out["mmap_read"]
    if not mm["completed"] or mm["oom_kill_events"] != 0:
        failures.append("mmap_read: reading a mapped file larger than the limit failed")
    if mm["peak_current_mib"] > limit:
        failures.append("mmap_read: memory use exceeded the limit")
    if mm["limit_hit_events"] < 1:
        failures.append("mmap_read: the limit was never reached, so nothing was evicted")

    pw = out["mmap_private_write"]
    if pw["completed"] or pw["killed_by_signal"] != 9 or pw["oom_kill_events"] < 1:
        failures.append("mmap_private_write: writing to a private mapping was not OOM-killed")
    return failures
