# Plan: 28 days through LLM inference infrastructure

One step a day, in the order of the twelve stages. Each day produces code or a model, tests, either
measured numbers or a clearly labelled simulation, and a write-up that includes what failed.

## The machine, and what that decides

Apple M3, 8 CPU cores, 10 GPU cores, 16 GB unified memory, macOS 26.2. No NVIDIA GPU.

That means roughly half the post's topics can be measured here and half cannot. Every day carries
one of these labels, and the label is what a resume claim is allowed to say.

| Label | Meaning | A resume may say |
|---|---|---|
| **M** | Measured on this machine, three or more runs, method documented | "benchmarked", with the hardware named |
| **S** | A simulation or analytical model. Calibrated against M numbers where stated, and tested | "modelled" or "simulated" |
| **L** | Study of specifications and documentation. Nothing run. | "studied", never a result |
| **G** | Measured on rented GPU hardware. Only with your explicit yes, and never unattended. | "benchmarked on" the named GPU |

Topics that need a data-centre GPU (CUDA and tensor cores, Hopper and Blackwell specifics, Nsight,
FlashAttention-3, NVLink, InfiniBand, multi-node serving) are **S or L by default**. A short
optional GPU block is described at the end. Nothing in this plan spends money without your yes.

## Facts to re-verify on the day

The tooling moves quickly and my knowledge has a cutoff. Each day's log starts by recording the
exact version or commit of every tool used. Features assumed below (llama.cpp prompt caching, KV
cache quantization, grammar-constrained decoding, draft-model speculation and RPC backend; MLX
quantization and speculative decoding; Triton's interpreter mode in a Linux container) must be
confirmed before use. If one is missing or broken, that is recorded as the day's result, and the
day falls back to the stated alternative.

## Stage 1: Systems foundations (Days 1 to 2)

- **Day 1: Benchmark harness and baseline.** *M.* A small Python client that measures
  time-to-first-token, inter-token latency, tokens per second and p50/p95/p99 against any
  OpenAI-compatible endpoint, with tests against a fake server. Build llama.cpp from source with
  Metal, run one small model, profile the harness itself (cProfile, py-spy) to bound its own
  overhead. Done when the harness numbers are reproducible within a stated spread.
- **Day 2: Memory hierarchy and Linux internals.** *M.* C microbenchmarks: sustained memory
  bandwidth, pointer-chasing latency at each cache level, naive vs blocked vs library matrix
  multiply. Model-file loading cold vs warm (mmap and the page cache). In a Linux container:
  page cache, cgroup memory limits and the OOM killer. Done when the numbers locate the cache
  and memory boundaries of this chip and the Linux behaviour is shown, not described.

## Stage 2: GPU architecture and modern silicon (Days 3 to 4)

- **Day 3: Roofline model.** *M and L.* A roofline calculator with specification tables for
  datacentre GPUs and this chip (vendor datasheet figures, each cited with the date fetched, all
  labelled L). Measure this chip's real bandwidth and matrix-multiply throughput with MLX and
  place the measured points on its roofline.
- **Day 4: GPU programming concepts on Metal.** *M and L.* Threadgroup-size and occupancy sweeps
  with a custom Metal kernel through MLX. A mapping from CUDA concepts (warp, SM, shared memory,
  tensor core) to what was measured. CUDA, `nvidia-smi` and DCGM stay L unless the GPU block runs.

## Stage 3: Transformer and MoE inference anatomy (Days 5 to 6)

- **Day 5: KV cache and memory calculator.** *M and S.* A calculator for KV cache and weight
  memory, validated against real model configurations (grouped-query attention, a multi-head
  latent attention model, a mixture-of-experts model) and against what llama.cpp reports.
- **Day 6: Prefill, decode and arithmetic intensity.** *M and S.* Predict decode tokens per second
  from bandwidth divided by bytes per token, and compare with measurements across models and
  quantizations. Show prefill is compute-bound. Time to first token against prompt length. A
  small mixture-of-experts router simulation showing load imbalance, plus a measurement on a
  small MoE model if it fits.

## Stage 4: Serving engines and constrained decoding (Days 7 to 9)

- **Day 7: Serving engines on this machine.** *M.* llama.cpp server under 1 to 16 concurrent
  clients with different parallel-slot settings, compared with Ollama and an MLX server. vLLM,
  SGLang, TGI and TensorRT-LLM need NVIDIA hardware and are **L**: a feature and design
  comparison from their documentation.
- **Day 8: Continuous batching and paged KV memory, from scratch.** *S.* A scheduler simulator
  and a block-based KV allocator, with tests, compared with static batching and contiguous
  allocation, driven by a trace calibrated from Day 7.
- **Day 9: Constrained decoding.** *M and S.* llama.cpp grammar and JSON-schema decoding: valid
  output rate and speed cost, with and without. A minimal logit-masking decoder over a toy
  tokenizer, with tests that every output parses.

## Stage 5: KV cache and memory hierarchies (Days 10 to 11)

- **Day 10: Prefix caching and chunked prefill.** *M and S.* Time to first token with and without
  a shared prompt prefix, and the effect of the micro-batch size. A simulation of chunked prefill
  protecting inter-token latency under mixed load.
- **Day 11: KV quantization, offload and eviction.** *M and S.* Memory, speed and quality of a
  quantized KV cache. Saving a session to SSD and restoring it, against recomputing it: where is
  the crossover. An eviction-policy simulator (LRU, LFU, prefix-aware, size-aware) on synthetic
  multi-turn traces.

## Stage 6: Quantization and compression (Days 12 to 13)

- **Day 12: The quantization ladder.** *M.* One model at several GGUF quantization levels: file
  size, perplexity on a public text sample (never committed), prefill and decode speed. A Pareto
  plot of quality against speed against size.
- **Day 13: Quantization internals.** *M and S.* Group-wise INT8 and INT4 quantizers written from
  scratch with error analysis, an activation-aware scaling toy in the spirit of AWQ, emulated
  FP8 and FP4 number formats, and MLX 4-bit against GGUF Q4. Real AWQ, GPTQ and FP4 pipelines are
  **L** or **G**.

## Stage 7: Speculative decoding and parallelism (Days 14 to 15)

- **Day 14: Speculative decoding.** *M and S.* The accept and reject algorithm from scratch, with
  a statistical test that the output distribution equals the target model's. Draft-model speedup
  and acceptance rate measured across draft sizes and temperatures.
- **Day 15: Parallelism.** *S and L.* Communication volume per layer per token for tensor,
  pipeline, expert and sequence parallelism, tested against hand-derived formulas, and when each
  becomes interconnect-bound for published link speeds. Medusa and EAGLE as a written analysis
  of acceptance length.

## Stage 8: Kernel-level optimization (Days 16 to 17)

- **Day 16: Attention from first principles.** *M and S.* Tiled online-softmax attention in
  NumPy, tested for exactness against naive attention, with memory-traffic accounting. Fused
  against unfused attention timed in MLX on this chip.
- **Day 17: Fusion and kernels.** *M and L.* Operator fusion measured with `mx.compile`, a custom
  Metal kernel against its unfused equivalent, each placed on the roofline. Triton is attempted
  in a Linux container in interpreter mode, and the outcome recorded either way. CUDA graphs,
  FlashAttention-3 and Nsight Compute are **L** unless the GPU block runs.

## Stage 9: Distributed and disaggregated inference (Days 18 to 19)

- **Day 18: Disaggregated prefill and decode.** *S.* A queueing simulation calibrated with
  measured per-token costs, tuning the prefill to decode ratio against latency targets, and KV
  transfer cost over published NVLink, InfiniBand and Ethernet speeds.
- **Day 19: What multi-node adds.** *S, L and a small M.* The llama.cpp RPC backend between two
  processes on this machine, measuring protocol overhead only (loopback is not a real fabric, and
  the write-up says so). An expert-parallel all-to-all cost model. A reading of one open-source
  disaggregated serving stack's architecture.

## Stage 10: Autoscaling, FinOps and economics (Days 20 to 21)

- **Day 20: Cold starts and queue-based scaling.** *M and S.* Measured model load and container
  start times. A local kind cluster with KEDA scaling a mock inference service whose service
  times come from Day 7, showing how cold-start time limits scale-out speed.
- **Day 21: Cost per token.** *M and S.* Cost per million tokens from measured throughput and
  hourly prices fetched from provider pages on the day (linked and dated, never from memory),
  plus MFU and memory-bandwidth utilization of the measured runs. Spot-instance orchestration is
  **L**.

## Stage 11: AI gateway, routing and reliability (Days 22 to 24)

- **Day 22: Gateway core.** *M.* Routing across local models by task, cost and latency, fallback
  chains and circuit breakers, with tests.
- **Day 23: Caching, limits and budgets.** *M.* Semantic caching with per-tenant keys (a shared
  cache is a data leak, the same lesson as the permission-filtered retrieval in the other lab),
  token-bucket rate limits, and per-tenant token budgets.
- **Day 24: SLOs, session routing and chaos.** *M.* Time-to-first-token and inter-token latency
  histograms against targets, session-sticky routing to keep prefix caches warm (measured with
  two server instances), and fault injection showing the effect on the targets.

## Stage 12: Edge inference and portfolio (Days 25 to 28)

- **Day 25: On-device benchmark matrix.** *M.* llama.cpp, MLX and Ollama across three models and
  several quantizations on one consistent method. WebGPU in a browser if it is feasible here.
- **Day 26: Write-ups and a reference deployment.** *M.* Three standalone write-ups taken from
  the measured results, and the kind, KEDA and gateway stack documented as a reference
  deployment with its limits and costs.
- **Day 27: Reproducibility and an honesty audit.** *M.* One command that regenerates the
  headline tables, CI that runs the tests, and a pass over every claim in the repository
  against the evidence ledger.
- **Day 28: Resume.** Draft resume bullets from the evidence ledger only, using the label rules
  above. You approve the wording before any resume file is changed.

## Optional GPU block

A few hours on one rented GPU would turn some **L** items into **G**: CUDA and tensor-core
microbenchmarks, `nvidia-smi` and DCGM readings, Nsight Compute on a real kernel, and vLLM or
SGLang against llama.cpp. I will check current prices and bring you a cap before any decision.
Until you say yes, none of it happens and the resume does not claim it.
