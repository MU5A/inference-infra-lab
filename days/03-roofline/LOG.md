# Day 3 log: roofline model

Appended as the work went. Times are local.

## 1. Start, 2026-11-23 about 11:00

- Machine: MacBook Air Mac15,12 (Apple M3, 4 performance and 4 efficiency cores, 10-core GPU
  by `system_profiler`, 16 GB, no fan), macOS 26.2, Python 3.12.8.
- Free disk 5.5 GiB at the start, below the runbook's 8 GB rule. The footprint of the day is about
  80 MB for MLX plus small JSON files, so you chose to proceed under a recorded exception
  (the runbook itself is unchanged). Free disk kept falling during the day (5.1 GiB after the MLX
  install, 3.8 GiB about 25 minutes later). This repository wrote about 80 MB of that; the rest
  was not investigated.
- On battery power at the start. The runbook's pre-flight refuses benchmark runs on battery, so no
  measurement was taken until the machine was on AC (see the end of this file for what happened).

## 2. MLX download (approved by you in chat before it started)

From PyPI, into a new empty scratch folder, checked against the size and sha256 PyPI publishes
for those exact files, then installed with `pip install --no-deps --no-index` into this
repository's `.venv` only.

| File | Size | sha256 |
|---|---|---|
| `mlx-0.32.4-cp312-cp312-macosx_26_0_arm64.whl` | 667,622 bytes | `7928869076db6267ab3ebdb253b2c80be1e738b3032c783170bd1d59d460c578` |
| `mlx_metal-0.32.4-py3-none-macosx_26_0_arm64.whl` | 78,106,861 bytes | `514df92d41a286895d2564b4b18f4da45096bd7a4a0c47e1042f345ad5a7f936` |

`mlx` requires `mlx-metal==0.32.4` on macOS (PyPI metadata), which is why there are two files.
Device check: `Device(gpu, 0)`, `device_name` Apple M3, `architecture` applegpu_g15g,
`memory_size` 17,179,869,184, `max_recommended_working_set_size` 12,713,115,648.

## 3. Specification figures (label L)

Method: each vendor page was fetched on 2026-11-23, converted to text, and read by hand (script in
the scratch folder, not committed). Where a table cell mattered, the raw HTML was read too.

Three pages were first read by subagents. Their reports were checked against the pages, and three
things were wrong or missing, which is why the committed file is built from the pages and not from
the reports:

- **A100 table misread.** The subagent read "156 TFLOPS | 312 TFLOPS*" as PCIe then SXM. The raw
  HTML shows one merged cell (`colspan="4"`) spanning both columns, and the page's note says
  "* With sparsity". So 156 is dense, 312 is with sparsity, and the same holds for BF16 and FP16
  (312 dense, 624 with sparsity). Taking the subagent's reading would have halved the A100's
  roof.
- **B200 per-GPU memory and bandwidth** were reported as not found. They are on NVIDIA's HGX
  reference-architecture page (180 GB HBM3e, "Up to 8TB/s" per GPU). The HGX product page prints
  compute only for the 8-GPU baseboard, so the per-GPU compute is derived by dividing by eight,
  and marked derived.
- **Apple M3 memory bandwidth** was reported as not found, because the pages the subagent read now
  describe the M5 generation. It is on the M3 MacBook Air tech specs page ("100GB/s memory
  bandwidth"). Neither that page nor Apple's 2023 M3 announcement prints any GPU or CPU FLOPS
  figure (checked by searching both for "flops"), so none is used: this chip's compute roofs come
  from measurement only.
- AMD's page timed out for the subagent twice. It was fetched directly without trouble.

Derivations in `specs/hardware_specs.json`, each re-computed by `tests/test_specs_file.py`:
H100 and H200 dense FP16 is half of the printed with-sparsity figure (989.5 TFLOPS); B200 dense
FP16 per GPU is the baseboard's 36 PFLOPS with sparsity, halved (NVIDIA's own footnote), divided
by 8 (2,250 TFLOPS). The A100 and MI300X dense figures are printed.

## 4. Order of work, so the predictions cannot have been fitted afterwards

1. `f6ee243`: the roofline calculator, the five predictions and the checks that judge them.
2. `a8dce0f`: the measurement worker, the gated runner, the analysis and the specification table.
3. A dry run of each worker once, un-gated, on battery power, to see that the code ran at all.
   It was after the commits above and before any measurement that counts. Its output was not saved
   and is not used. I did see it: it showed streaming near 70 to 90 GB/s and matrix multiply
   between 2 and 3 TFLOPS, so the predictions were not blind to the order of magnitude in the
   way they were when written. Only the gated, three-run results below are evidence.

## 5. Measurements, 2026-11-23 evening

You plugged the Mac in and said the machine was quiet. It was on AC power at 100%. Chrome (about
25% of a core), a video player (VLC, about 15%), WindowServer and `coreaudiod` were still running,
and the pre-flight, which refuses on load, free memory, GPU use and battery, passed.

**Run 1, `python scripts/day03_run.py`** (result files written 22:09:54, 22:10:15 and 22:10:38):

- Bandwidth: held. Stable (variation 1.1% to 2.5%), validated, decode probe 154.7 before and 157.3
  after.
- Matrix multiply: held. Stable (variation at most 2.8%), every product validated against NumPy,
  probe 156.9 before and 151.9 after.
- Decode sweep: did not hold. Batches 64, 128, 256 and 512 varied 19.7%, 25.3%, 21.0% and 10.1%
  across the three runs against a 10% limit. The third run was the slow one. The sampler recorded
  `ANECompilerService` (an Apple system process running as root) at up to 89.8% of a core. I moved
  the file to `results/rejected/gpu_skinny_attempt1_unstable.json`.

**Attempt 2, `python scripts/day03_run.py --only skinny --wait-quiet 900`** (file written 22:12:25,
1 minute 47 seconds after attempt 1's). The quiet-wait passed at once, so this ran with almost no
rest. No contention flag, probe 158.0 before and 155.5 after, but batches 64, 128 and 256 varied
10.6%, 11.1% and 10.4%, and in each the runs fell one after another. Rejected.

**Decision, made after seeing both failures:** add an optional `--rest SECONDS` to the runner
(commit 5dc6914), idling before each of the three runs. I did not change the 10% limit or the
analysis. The bandwidth and matrix-multiply phases ran without rests and held, so they were not
re-run; the rest applies only to the decode sweep, and the day's README says so. I read no
temperature or clock, so the claim that the slowdown is thermal is a hypothesis, not a finding.

**Attempt 3, `python scripts/day03_run.py --only skinny --rest 240 --wait-quiet 300`** (file
written 22:27:20, exit 0). Held: stable, validated, no drift (probe 154.8 before, 158.4 after), and
every batch repeated within 1.0%. Compute-bound batches sit at 3.01 TFLOPS in all three runs.

**The contended flag on attempt 3.** The sampler reported a peak of 263.5% of a core for all other
processes and a one-minute load of 5.9, with Chrome, `bird`, `duetexpertd` and
`IntelligencePlatformComputeService` as the busiest. My `--rest` sleeps sit inside the sampler's
window (359 samples at 2 seconds each, about 12 minutes), so the flag cannot tell whether any burst
overlapped the timed work, which is a small part of it (the first run's three phases took about 20
seconds each, probes included, judging by the result files' times). The sampler's own documentation says variation decides stability
and a contended run is "looser, not wrong". The runs agree within 1.0%, so I cite the file and
disclose this.

**GPU use by other processes** was checked only by the pre-flight at the start of each phase
(refuses a peak above 25% or a mean above 5%), not during the runs; the sampler is CPU only. The
GPU readings in the result files (peak 64% to 98%, mean 13% to 20%) were taken right after the
final decode probe, so they cannot separate that probe's use from anyone else's. This is under
"Limits" in the README.

## 6. Report, chart, and what looking at it found

`python scripts/day03_report.py` wrote the tables, the analysis and the chart. All five
predictions held. I then rendered the chart and looked at it, which the tables had not made me do:
the `add` kernel's point (7 GFLOPS) was under the lowest axis line, and the rotated axis title
overlapped the tick labels. I lowered the floor from 10 to 1 GFLOPS, widened both margins, and added
a test that the lowest measured point is inside the plot, a test that the longest legend label
fits (6.6 pixels per character is a generous estimate; Helvetica at 12 px measures the longest at
176 pixels, ending at x=918 of 980), and a test that every point in `analysis.json` is inside the
chart range. QuickLook crops a landscape SVG to a square, which made the legend look cut off; that
was the renderer, not the file.

## 7. A CI failure caught before pushing

CI runs `ruff format --check .` and `scripts/mutation_check.py` had been committed (aa64500) without
the formatter. `ruff format` changed quote style only; I compared the parsed programs of the old and
new file and they are identical. Fixed in 5dc6914. Nothing had been pushed.

## 8. Write-up

`days/03-roofline/README.md` was written from a template with the two generated table files
substituted in verbatim (the template is not committed), and `tests/test_day_docs.py` now checks
both tables appear in it unchanged. Free disk at this point: 3.0 GiB, under the runbook's 8 GB rule
(exception approved by you at the start of the day).
