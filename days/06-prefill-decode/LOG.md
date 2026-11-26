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
