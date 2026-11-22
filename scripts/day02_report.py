"""Render Day 2 results as Markdown tables, straight from the results files.

No number in the write-up is typed by hand: they all come from this output, and a test checks that
the day's README contains it verbatim.
"""

import json
import pathlib

from inferlab.cgroup import DIRTY_WRITE_MIB

DAY = pathlib.Path(__file__).resolve().parents[1] / "days" / "02-memory-hierarchy"


def span(values, digits=2):
    lo, hi = min(values), max(values)
    fmt = f"{{:.{digits}f}}"
    return fmt.format(lo) if lo == hi else f"{fmt.format(lo)} to {fmt.format(hi)}"


def render_linux(report: dict) -> str:
    runs = report["runs"]
    n = len(runs)
    anon = [r["anonymous"] for r in runs]
    mm = [r["mmap_read"] for r in runs]
    pw = [r["mmap_private_write"] for r in runs]
    dirty = report["dirty_write"]
    limit = runs[0]["info"]["limit_mib"]

    def count(flags):
        return f"{sum(flags)} of {n}"

    killed = count(a["killed_by_signal"] == 9 for a in anon)
    anon_at = span([a["last_progress_mib"] for a in anon], 0)
    anon_hits = span([a["limit_hit_events"] for a in anon], 0)
    mapped_ok = count(m["completed"] and m["oom_kill_events"] == 0 for m in mm)
    mapped_peak = span([m["peak_current_mib"] for m in mm])
    mapped_hits = span([m["limit_hit_events"] for m in mm], 0)
    write_killed = count(w["killed_by_signal"] == 9 for w in pw)
    write_at = span([w["last_progress_mib"] for w in pw], 0)
    unsynced = count(d["unsynced"]["oom_killed"] for d in dirty)
    unsynced_exit = span([d["unsynced"]["exit_code"] for d in dirty], 0)
    synced = count(not d["synced"]["oom_killed"] and d["synced"]["exit_code"] == 0 for d in dirty)
    synced_exit = span([d["synced"]["exit_code"] for d in dirty], 0)

    outcomes = [
        "| Experiment | Outcome | Detail (range over the runs) |",
        "|---|---|---|",
        f"| Ordinary memory allocated until the limit | OOM-killed ({killed}) "
        f"| killed after {anon_at} MiB allocated; the limit was reached {anon_hits} times |",
        f"| Read a memory-mapped {mm[0]['file_mib']} MiB file, twice the limit "
        f"| completed, no kill ({mapped_ok}) "
        f"| total memory peaked at {mapped_peak} MiB against a limit of {limit:.0f}; "
        f"the limit was reached {mapped_hits} times |",
        f"| Write to a privately mapped file | OOM-killed ({write_killed}) "
        f"| killed after {write_at} MiB written |",
        f"| Write a {DIRTY_WRITE_MIB} MiB file without fsync | OOM-killed ({unsynced}) "
        f"| Docker reports OOMKilled, exit code {unsynced_exit} |",
        f"| The same write with fsync after each chunk | completed ({synced}) "
        f"| exit code {synced_exit} |",
    ]

    def stage(name):
        return span([r["page_cache"]["stages"][name]["file_mib"] for r in runs])

    size = runs[0]["page_cache"]["file_mib"]
    cache_secs = span([r["page_cache"]["read_seconds"]["from_cache"] for r in runs], 4)
    drop_secs = span([r["page_cache"]["read_seconds"]["after_drop"] for r in runs], 4)
    cache = [
        f"Page cache accounting for a {size} MiB file "
        "(memory the cgroup attributes to cached file data):",
        "",
        "| Stage | File memory, MiB |",
        "|---|---|",
        f"| Before | {stage('start')} |",
        f"| After writing the file | {stage('after_write')} |",
        f"| After reading it | {stage('after_read')} |",
        f"| After asking the kernel to drop it (fadvise DONTNEED) | {stage('after_drop')} |",
        f"| After reading it again | {stage('after_cold_read')} |",
        "",
        f"Reading the file took {cache_secs} s from the cache and {drop_secs} s after it was "
        "dropped.",
    ]
    image, digest = report["image"].split("@")
    head = (
        f"Image `{image}@{digest[:19]}...`, Docker {report['docker']}, memory limit "
        f"{report['limit']} with no swap, {n} fresh containers. "
        f"Every prediction written before the first run held: {report['all_predictions_held']}."
    )
    return "\n\n".join([head, "\n".join(outcomes), "\n".join(cache)])


def main() -> None:
    results = DAY / "results"
    linux = results / "linux_memory.json"
    if linux.exists():
        text = render_linux(json.loads(linux.read_text()))
        (results / "linux_table.md").write_text(text + "\n")
        print(text)


if __name__ == "__main__":
    main()
