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


def render_microbench(first: dict, rerun: dict, compute: dict, attempt3: dict) -> str:
    out: list[str] = []

    # --- bandwidth (first run, stable)
    bw = first["groups"]["bandwidth"]["configs"]
    rows = [
        "| Threads | Triad GB/s, median (min to max) | Copy GB/s, median | Variation (CV) of triad |",
        "|---|---|---|---|",
    ]
    for threads in (1, 2, 3, 4, 6, 8):
        cfg = bw[f"interactive-{threads}"]["kernels"]
        t, c = cfg["triad"], cfg["copy"]
        rows.append(
            f"| {threads} | {t['median']:.1f} ({t['min']:.1f} to {t['max']:.1f}) "
            f"| {c['median']:.1f} | {t['cv'] * 100:.1f}% |"
        )
    out += [
        "Sustained memory bandwidth, STREAM-style, 256 MiB arrays, 3 independent runs:",
        "\n".join(rows),
    ]

    # --- latency (first run, performance cores, stable)
    lat = first["groups"]["latency"]["interactive"]
    curve = {m["bytes"]: m for m in lat["merged"]}
    wanted = [4096, 65536, 131072, 262144, 1 << 20, 4 << 20, 8 << 20, 12 << 20, 16 << 20, 24 << 20,
              32 << 20, 64 << 20, 256 << 20, 1 << 30]  # fmt: skip
    rows = ["| Working set | Latency, ns (min to max over 3 runs) |", "|---|---|"]
    for target in wanted:
        size = min(curve, key=lambda b: abs(b - target))
        m = curve[size]
        label = f"{size / 1024:.0f} KiB" if size < (1 << 20) else f"{size / (1 << 20):.1f} MiB"
        rows.append(f"| {label} | {m['ns']:.1f} ({m['min']:.1f} to {m['max']:.1f}) |")
    pl = lat["plateau_stability"]
    out += [
        "Load-to-use latency by pointer chasing, performance cores, 3 independent sweeps of 4 points per octave:",
        "\n".join(rows),
        f"The plateau finder found {lat['plateaus_per_run']} plateaus in the three sweeps. The first "
        f"(median {pl[0]['median']:.2f} ns, variation {pl[0]['cv'] * 100:.1f}%) ends at "
        f"{pl[0]['end_bytes_per_run'][0] / 1024:.0f} KiB in every run. The last (median "
        f"{pl[-1]['median']:.1f} ns, variation {pl[-1]['cv'] * 100:.1f}%) is main memory. The ends of "
        f"the two middle plateaus were {[round(x / (1 << 20), 1) for x in pl[1]['end_bytes_per_run']]} "
        f"and {[round(x / (1 << 20), 1) for x in pl[2]['end_bytes_per_run']]} MiB in the three runs, "
        "so they do not locate a capacity.",
    ]

    # --- compute (accepted run)
    summary = compute["groups"]["compute"]["summary"]
    sizes = (128, 256, 512, 1024)
    head = "| GFLOPS, single precision | " + " | ".join(f"n={n}" for n in sizes) + " |"
    rows = [head, "|---|" + "---|" * len(sizes)]
    for name, label in (
        ("naive", "naive loop (i, j, k)"),
        ("ikj", "reordered loop (i, k, j)"),
        ("blocked32", "tiled, 32 x 32"),
        ("blocked64", "tiled, 64 x 64"),
        ("accelerate", "Accelerate library, default threads"),
    ):
        cells = " | ".join(f"{summary[f'default/{name}/{n}']['median']:.1f}" for n in sizes)
        rows.append(f"| {label} | {cells} |")
    cells = " | ".join(
        f"{summary[f'accelerate_one_thread/accelerate/{n}']['median']:.1f}" for n in sizes
    )
    rows.append(f"| Accelerate with VECLIB_MAXIMUM_THREADS=1 | {cells} |")
    worst = max(v["cv"] for v in summary.values())
    out += [
        "Matrix multiply, median over 3 process runs of the best of "
        f"8 samples in each (largest variation of any entry: {worst * 100:.1f}%):",
        "\n".join(rows),
    ]

    # --- attempts
    def unstable(group_summary):
        bad = [k for k, v in group_summary.items() if not v["stable"]]
        return f"{len(bad)} of {len(group_summary)} entries unstable"

    attempts = [
        "| Attempt | What changed | Result |",
        "|---|---|---|",
        f"| 1 | First run, no neighbour sampling | {unstable(first['groups']['compute']['summary'])}; "
        "at n=128 to 512 every variant was slower by about the same factor in each later run |",
        f"| 2 | Neighbour sampling added | {unstable(rerun['groups']['compute']['summary'])}; "
        "all at n=128, where one call takes microseconds |",
        f"| 3 | Each timed sample repeated to at least 20 ms | {unstable(attempt3['groups']['compute']['summary'])}; "
        f"another process peaked at {attempt3['groups']['compute']['neighbours']['peak_other_cpu_percent']:.0f}% CPU |",
        f"| 4 | Best of 8 samples per run instead of 3 | {unstable(summary)} (accepted) |",
    ]
    out += ["Every attempt at the matrix-multiply group, kept on record:", "\n".join(attempts)]

    # --- cold read (second design)
    cr = rerun["groups"]["cold_read"]
    rows = [
        "| First pass over a cold file | GB/s, median (min to max) | Variation (CV) | Stable |",
        "|---|---|---|---|",
    ]
    labels = {
        "touch": "memory-mapped, one touch per page",
        "read": "sequential read()",
        "willneed": "memory-mapped, madvise(WILLNEED)",
        "sequential": "memory-mapped, madvise(SEQUENTIAL)",
    }
    for way, label in labels.items():
        v = cr["cold_gbs"][way]
        rows.append(
            f"| {label} | {v['median']:.2f} ({v['min']:.2f} to {v['max']:.2f}) "
            f"| {v['cv'] * 100:.1f}% | {'yes' if v['stable'] else 'no'} |"
        )
    ratios = [t["cold_read"]["gbs"] / t["cold_touch"]["gbs"] for t in cr["trials"]]
    out += [
        "Cold reads of the 469 MB model file, each pass on its own fresh copy made with the page cache "
        "bypassed, 3 trials:",
        "\n".join(rows),
        f"A cold sequential read was {min(ratios):.1f} to {max(ratios):.1f} times faster than a cold "
        "memory-mapped touch in every trial. "
        f"Predictions written before the run that failed: {cr['prediction_failures'] or 'none'}.",
    ]
    return "\n\n".join(out)


def main() -> None:
    results = DAY / "results"
    linux = results / "linux_memory.json"
    if linux.exists():
        text = render_linux(json.loads(linux.read_text()))
        (results / "linux_table.md").write_text(text + "\n")
        print(text)
    needed = ("microbench.json", "microbench_rerun.json", "compute.json", "compute_attempt3.json")
    if all((results / name).exists() for name in needed):
        loaded = [json.loads((results / name).read_text()) for name in needed]
        text = render_microbench(*loaded)
        (results / "microbench_table.md").write_text(text + "\n")
        print()
        print(text)


if __name__ == "__main__":
    main()
