# Day 6 log: prefill, decode and arithmetic intensity

Appended as the work goes. Times are local (WAT).

## 0. Start, 2026-11-26

- The unattended 09:07 run stopped at its pre-flight disk check (5.2 GiB free) and wrote
  `STUCK.md`. By about 10:13 free disk was down to about 1 GB with 8 GB of swap files. The owner
  cleared nothing by hand; I cleared the Go build and npm caches (under the owner's earlier
  approval), and the owner then restarted the Mac.
- 10:53, nine minutes after the restart: 12.40 GB free, no swap, AC power at 100%,
  `pmset -g therm` with no thermal or performance warning, load average 4.45, 9.07, 6.87. Docker
  was not running. Chrome (2.11 GiB) and Slack (0.82 GiB) had reopened.
- This day is run by hand with the owner present, not unattended.

### Downloads, approved by the owner

The owner approved three files after being shown their source, size and licence (reply: "yes").
All three are from Qwen's own repositories on Hugging Face, ungated, `apache-2.0` as listed:

| File | Repository | Repository commit at the time | Listed size |
|---|---|---|---|
| `qwen2.5-0.5b-instruct-q8_0.gguf` | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` | `9217f5db79a29953eb74d5343926648285ec7e67` | 0.68 GB |
| `qwen2.5-0.5b-instruct-fp16.gguf` | same | same | 1.27 GB |
| `qwen2.5-1.5b-instruct-q4_k_m.gguf` | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | `91cad51170dc346986eccefdc2dd33a9da36ead9` | 1.12 GB |

Fetched with `scripts/fetch_model.sh`, which refuses a file whose SHA-256 differs from the digest
the repository publishes. The Day 1 file (`...0.5b-instruct-q4_k_m.gguf`, sha256 74a4da8c...) is
the same digest the first repository lists for it, so the four files share a source.

A mixture-of-experts model was left out on purpose: the smallest OLMoE file listed is 2.56 GB, and
adding it would have broken the runbook's rule that a day uses at most a third of the free disk.
The router simulation stands in for it and is labelled S.

### Design, committed before any timing

- The predictions are in the docstring of `src/inferlab/decode_model.py`, with thresholds as
  constants there. They were committed before any Day 6 timing and before the headers of the three
  new files were read. Day 1's figures for the 0.5B Q4_K_M file were already known, so anything
  about that file alone is not blind; predictions 8 and 9 were hand-estimated.
- One `llama-bench` smoke run (8 prompt tokens, 4 generated) was done to learn the JSON field
  names; the script that read it printed only the names and which depths appeared, no timing.
- Stability, as on Day 3: a figure is cited only if three independent process runs agree with a
  coefficient of variation of at most 10%; a prediction that needs an unstable figure is not judged.
- Every timed process is bracketed by Day 1's decode probe (refused if more than 20% below the
  baseline), after a rest. Rounds interleave the four models.

## 1. Measurements, scheduled run of 2026-11-27 (nobody present)

- Pre-flight, 2026-11-27: `df -h .` showed 10 GiB free (95% used); `gh auth status` logged in as
  MU5A; git clean apart from the untracked `STUCK.md` left by the 2026-11-26 09:07 stop (not
  committed, deleted when the day completes); AC power, battery 100% charged; `pmset -g therm`: no
  thermal or performance warning; load average 2.31, 2.49, 2.32; no swap in use; 69% of memory
  free. Docker Desktop was not running (only the `com.docker.vmnetd` helper, which always runs).
- No download was needed: the four model files approved and fetched on 2026-11-26 are in
  `models/`. The design, the runner and the ten predictions are unchanged since 69f5903.
- Bench phase: `PYTHONPATH=scripts:src python scripts/day06_run.py --out
  days/06-prefill-decode/results --phase bench`, console output in `bench.out` (ignored by git).
- 09:38 to 09:50: bench phase, exit 0, `results/bench.json`. Console lines, one per process
  (median of 3 in-process repetitions, tokens/s):
  - round 0: 0.5B Q4_K_M tg@0 159.8, tg@4096 145.3; Q8_0 140.8, 129.2; FP16 85.0, 81.4; 1.5B 80.4, 72.6
  - round 1: 166.2, 149.1; 144.4, 131.4; 84.9, 81.5; 80.0, 72.5
  - round 2: 138.7, **103.8**; 143.8, 122.9; 85.9, 81.4; 78.8, 72.2
  - every probe passed (no drift). The after-probe of every round 2 process was lower (144.7 to
    152.7 against 164.1 to 166.1 before). Round 0 of the 0.5B Q4_K_M ran beside a process listed
    as `deleted` at 96.7% CPU and was marked contended; kept, as the protocol says.
- Stability: 15 of 16 figures within 10%. The 0.5B Q4_K_M `tg@4096` varied by 15.5% (round 2's
  103.8). Not rerun: choosing to repeat after seeing a result would bias it. Prediction 6 is
  therefore not judged.
- 09:50 to 09:53: `--phase ttft`, exit 0, `results/ttft.json`. Medians (ms) at 40, 257 and 1,469
  tokens: 0.5B Q4_K_M 25.3, 93.8, 500.4; Q8_0 23.8, 85.5, 455.2; FP16 24.6, 83.6, 434.6; 1.5B
  73.1, 297.3, 1546.6. No request saw a cached token; all probes passed.
- `python scripts/day06_report.py`: 8 of 10 predictions held. Prediction 2 failed (0.5B FP16 at
  0.964 of its bound, limit 0.95). Prediction 6 not judged (above).
- Surprise: across the 0.5B files prefill runs in the reverse order of decode (FP16 3590,
  Q8_0 3398, Q4_K_M 3170 tokens/s). Not predicted; cause not measured.
- Added `tests/test_day6_results.py` (recomputes the analysis and tables from the committed files,
  pins the verdicts and the one unstable figure), the Day 6 check in `tests/test_day_docs.py`, and
  26 mutation entries for `decode_model.py`, `decode_report.py` and `moe_router.py`.
- `ruff check .`, `ruff format --check .`: clean. `pytest -q`: 596 passed.
