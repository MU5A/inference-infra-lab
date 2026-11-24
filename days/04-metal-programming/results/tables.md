#### What the kernels reported about how they ran

| Phase | Configurations | Ran at the threadgroup size asked, in every run | SIMD-group width reported |
|---|---|---|---|
| stream | 12 | 12 of 12 | 32 |
| occupancy | 6 | 6 of 6 | 32 |
| fma | 12 | 12 of 12 | 32 |

#### Streaming triad by threadgroup size

| Threadgroup size | scalar, GB/s median (min to max) | scalar, % of MLX triad | vec4, GB/s median (min to max) | vec4, % of MLX triad |
|---|---|---|---|---|
| 32 | 84.2 (83.7 to 85.3) | 97% | 83.7 (82.6 to 90.8) | 96% |
| 64 | 88.7 (86.0 to 101.3) | 102% | 83.2 (82.9 to 84.3) | 95% |
| 128 | 86.4 (84.4 to 89.0) | 99% | 84.1 (83.9 to 84.4) | 96% |
| 256 | 86.4 (84.4 to 87.0) | 99% | 82.2 (81.4 to 85.4) | 94% |
| 512 | 98.6 (90.2 to 102.9) | 113% | 84.5 (83.3 to 84.9) | 97% |
| 1024 | 87.9 (85.8 to 89.2) | 101% | 83.3 (74.0 to 83.8) | 96% |

#### Threadgroup memory and bandwidth

| Threadgroup memory per group of 256 | GB/s median (min to max) | Variation | Relative to 1024 bytes |
|---|---|---|---|
| 1 KiB | 85.5 (83.7 to 86.1) | 1.2% | 1.00 |
| 4 KiB | 84.7 (80.4 to 85.2) | 2.6% | 0.99 |
| 8 KiB | 81.4 (77.8 to 82.6) | 2.5% | 0.95 |
| 16 KiB | 93.2 (91.2 to 93.3) | 1.1% | 1.09 |
| 24 KiB | 92.6 (90.7 to 93.2) | 1.1% | 1.08 |
| 32 KiB | 80.0 (78.6 to 81.0) | 1.3% | 0.94 |

#### Fused multiply-add throughput by threadgroup size

| Threadgroup size | fp32 TFLOPS, median (min to max) | fp32 variation | fp16 TFLOPS, median (min to max) | fp16 variation |
|---|---|---|---|---|
| 32 | 2.83 (2.83 to 2.84) | 0.1% | 3.06 (3.05 to 3.06) | 0.1% |
| 64 | 2.85 (2.83 to 2.85) | 0.2% | 3.06 (3.05 to 3.06) | 0.1% |
| 128 | 2.84 (2.84 to 2.85) | 0.1% | 3.06 (3.05 to 3.06) | 0.2% |
| 256 | 2.85 (2.83 to 2.85) | 0.2% | 3.06 (3.05 to 3.06) | 0.1% |
| 512 | 2.84 (2.84 to 2.84) | 0.0% | 3.06 (3.06 to 3.06) | 0.0% |
| 1024 | 2.84 (2.84 to 2.85) | 0.1% | 3.06 (3.05 to 3.06) | 0.2% |

#### Predictions

| Prediction | Result |
|---|---|
| Best hand-written triad at least 70% of Day 3's MLX triad; threadgroup sizes 64 to 1024 within 10% of each variant's best | did not hold: scalar threadgroup 64: 90% of the variant's best, more than 10% below it; scalar threadgroup 128: 88% of the variant's best, more than 10% below it; scalar threadgroup 256: 88% of the variant's best, more than 10% below it; scalar threadgroup 1024: 89% of the variant's best, more than 10% below it |
| 32 KiB of threadgroup memory costs at least 20% of the 1 KiB bandwidth; no step up in memory more than 10% faster | did not hold: 32768 bytes of threadgroup memory cost 6% of the bandwidth at 1024 bytes, less than the predicted 20%; going from 8192 to 16384 bytes made it 14% faster, more than the 10% allowed |
| Best fp32 FMA between 2.5 and 4.5 TFLOPS and at least 0.95 times Day 3's fp32 matrix multiply; fp16 between 0.8 and 1.25 times fp32 | held |

#### Machine speed before and after each phase

| Phase | Decode probe before, tokens/s | Decode probe after, tokens/s |
|---|---|---|
| stream | 155.6 | 155.8 |
| occupancy | 156.1 | 157.3 |
| fma | 161.9 | 155.1 |
