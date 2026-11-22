# Rejected results: do not cite

These files came from measurement attempts that were disturbed by something else on the machine.
They are kept as evidence of what interference looks like, and so that nothing is quietly dropped.
**None of their numbers may be used in a claim.** The only Day 1 results that may be cited are in
`../baseline_llama_server.json`.

| File | What was wrong |
|---|---|
| `warm_cache.json` | Taken with the pre-flight clean at the start, but decode varied by up to 6.9% (one sample at 127.9 against 150 elsewhere), so something interfered mid-run. The size of the effect, a prompt cache cutting time to first token from tens or hundreds of milliseconds to about 10, is far larger than that noise. The exact numbers are not citable. |
| `llama_bench.json`, `llama_bench.environment.json` | Decode at 60.5 tokens/s with a standard deviation of 50.2 (samples 148.9, 36.1, 48.4, 43.8, 25.4) while a video player used the GPU. The pre-flight at the time looked only at CPU load and could not see it. An earlier, better llama-bench run was overwritten before it was saved to a committed file, so no raw data exists for it. |
| `harness_overhead.json`, `harness_overhead_profile.txt` | Three runs began with a clean pre-flight, but the load average was 6.4 by the end. Tails were large (p95 3 to 11 ms) and two earlier discarded runs had smaller ones, so load average did not predict the tails. The measurement has to be repeated under conditions that are verified throughout, not only at the start. |
