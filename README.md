# inference-infra-lab

A 28-day, one-step-a-day study of LLM inference infrastructure, done on a single Apple M3 laptop
with 16 GB of memory. Each day has code, tests, measured numbers or a labelled model, and a
write-up that includes what failed.

## What this is not

- Not production GPU serving experience, and not a cluster.
- Topics that need a data-centre GPU (CUDA and tensor cores, Hopper and Blackwell, Nsight,
  FlashAttention-3, NVLink, InfiniBand, multi-node serving) are modelled or studied here, and
  labelled that way, unless a day says it was measured on rented hardware.

## Evidence labels

| Label | Meaning |
|---|---|
| **M** | Measured on this machine, three or more runs, method documented |
| **S** | Simulation or analytical model, tested, calibrated where stated |
| **L** | Study of specifications and documentation, nothing run |
| **G** | Measured on rented GPU hardware (only if a day says so) |

Every claim is listed in [EVIDENCE.md](EVIDENCE.md) with the file that supports it.

## Status

| Day | Stage | Topic | Label | Status |
|---|---|---|---|---|
| 1 | 1. Systems foundations | Benchmark harness and baseline | M | done |
| 2 | 1. Systems foundations | Memory hierarchy and Linux internals | M | done |
| 3 | 2. GPU architecture and modern silicon | Roofline model | M, L | done |
| 4 | 2. GPU architecture and modern silicon | GPU programming concepts on Metal | M, L | done |
| 5 | 3. Transformer and MoE inference anatomy | KV cache and memory calculator | M, S | planned |
| 6 | 3. Transformer and MoE inference anatomy | Prefill, decode and arithmetic intensity | M, S | planned |
| 7 | 4. Serving engines and constrained decoding | Serving engines on this machine | M | planned |
| 8 | 4. Serving engines and constrained decoding | Continuous batching and paged KV memory, from scratch | S | planned |
| 9 | 4. Serving engines and constrained decoding | Constrained decoding | M, S | planned |
| 10 | 5. KV cache and memory hierarchies | Prefix caching and chunked prefill | M, S | planned |
| 11 | 5. KV cache and memory hierarchies | KV quantization, offload and eviction | M, S | planned |
| 12 | 6. Quantization and compression | The quantization ladder | M | planned |
| 13 | 6. Quantization and compression | Quantization internals | M, S | planned |
| 14 | 7. Speculative decoding and parallelism | Speculative decoding | M, S | planned |
| 15 | 7. Speculative decoding and parallelism | Parallelism | S, L | planned |
| 16 | 8. Kernel-level optimization | Attention from first principles | M, S | planned |
| 17 | 8. Kernel-level optimization | Fusion and kernels | M, L | planned |
| 18 | 9. Distributed and disaggregated inference | Disaggregated prefill and decode | S | planned |
| 19 | 9. Distributed and disaggregated inference | What multi-node adds | S, L, M | planned |
| 20 | 10. Autoscaling, FinOps and economics | Cold starts and queue-based scaling | M, S | planned |
| 21 | 10. Autoscaling, FinOps and economics | Cost per token | M, S | planned |
| 22 | 11. AI gateway, routing and reliability | Gateway core | M | planned |
| 23 | 11. AI gateway, routing and reliability | Caching, limits and budgets | M | planned |
| 24 | 11. AI gateway, routing and reliability | SLOs, session routing and chaos | M | planned |
| 25 | 12. Edge inference and portfolio | On-device benchmark matrix | M | planned |
| 26 | 12. Edge inference and portfolio | Write-ups and a reference deployment | M | planned |
| 27 | 12. Edge inference and portfolio | Reproducibility and an honesty audit | M | planned |
| 28 | 12. Edge inference and portfolio | Resume | none | planned |

See [PLAN.md](PLAN.md) for what each day builds and measures, [DAILY.md](DAILY.md) for how each
day is run, and [EVIDENCE.md](EVIDENCE.md) for every claim and what supports it.
