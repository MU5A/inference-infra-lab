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

Not started. See [PLAN.md](PLAN.md) for the 28 days and [DAILY.md](DAILY.md) for how each day is
run.
