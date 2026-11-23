#### Roofs

| Roof | Peak TFLOPS | Bandwidth GB/s | Ridge, FLOP/byte | Where it comes from |
|---|---|---|---|---|
| GPU, half precision | 3.08 | 90.9 | 33.9 | measured today through MLX |
| GPU, single precision | 2.78 | 90.9 | 30.5 | measured today through MLX |
| CPU, single precision | 1.65 | 85.1 | 19.4 | measured on Day 2 |

#### Streaming bandwidth on the GPU

| Kernel | Bytes moved per element | GB/s, median (min to max) | Variation |
|---|---|---|---|
| scale | 8 | 86.7 (83.7 to 87.0) | 1.7% |
| add | 12 | 85.4 (83.6 to 85.5) | 1.1% |
| triad | 12 | 87.2 (83.9 to 89.2) | 2.5% |
| sum | 4 | 90.9 (90.7 to 94.4) | 1.8% |

#### Matrix multiply on the GPU

| TFLOPS, median (min to max) | n=512 | n=1024 | n=2048 | n=4096 |
|---|---|---|---|---|
| fp32 | 2.02 (2.01 to 2.03) | 2.78 (2.77 to 2.78) | 2.73 (2.72 to 2.79) | 2.59 (2.52 to 2.63) |
| fp16 | 2.24 (2.15 to 2.24) | 3.07 (2.95 to 3.08) | 3.08 (2.99 to 3.09) | 3.05 (2.88 to 3.06) |
| bf16 | 2.24 (2.21 to 2.24) | 3.08 (3.06 to 3.08) | 3.09 (3.08 to 3.09) | 3.06 (3.02 to 3.06) |

#### Decode-shaped products on the GPU

| Batch | Intensity, FLOP/byte | TFLOPS | Weight traffic, GB/s | Fraction of roofline | Bound |
|---|---|---|---|---|---|
| 1 | 1 | 0.09 | 90.1 | 99% | memory |
| 2 | 2 | 0.17 | 83.1 | 91% | memory |
| 4 | 4 | 0.32 | 81.0 | 89% | memory |
| 8 | 7.98 | 0.64 | 80.3 | 88% | memory |
| 16 | 15.9 | 1.31 | 82.2 | 90% | memory |
| 32 | 31.8 | 1.48 | 46.5 | 51% | memory |
| 64 | 63 | 3.01 | 47.8 | 98% | compute |
| 128 | 124 | 3.01 | 24.3 | 98% | compute |
| 256 | 241 | 3.01 | 12.5 | 98% | compute |
| 512 | 455 | 3.02 | 6.6 | 98% | compute |
| 1024 | 819 | 3.01 | 3.7 | 98% | compute |

#### Every measured point against its roofline

| Series | Point | Intensity, FLOP/byte | Achieved GFLOPS | Roofline GFLOPS | Fraction of roofline | Bound |
|---|---|---|---|---|---|---|
| stream | scale | 0.125 | 11 | 11 | 95% | memory |
| stream | add | 0.0833 | 7 | 8 | 94% | memory |
| stream | triad | 0.167 | 15 | 15 | 96% | memory |
| stream | sum | 0.25 | 23 | 23 | 100% | memory |
| matmul fp32 | fp32-512 | 85.3 | 2016 | 2775 | 73% | compute |
| matmul fp32 | fp32-1024 | 171 | 2775 | 2775 | 100% | compute |
| matmul fp32 | fp32-2048 | 341 | 2733 | 2775 | 98% | compute |
| matmul fp32 | fp32-4096 | 683 | 2590 | 2775 | 93% | compute |
| matmul fp16 | fp16-512 | 171 | 2236 | 3084 | 73% | compute |
| matmul fp16 | fp16-1024 | 341 | 3072 | 3084 | 100% | compute |
| matmul fp16 | fp16-2048 | 683 | 3084 | 3084 | 100% | compute |
| matmul fp16 | fp16-4096 | 1.37e+03 | 3051 | 3084 | 99% | compute |
| matmul bf16 | bf16-512 | 171 | 2236 | 3084 | 72% | compute |
| matmul bf16 | bf16-1024 | 341 | 3082 | 3084 | 100% | compute |
| matmul bf16 | bf16-2048 | 683 | 3087 | 3084 | 100% | compute |
| matmul bf16 | bf16-4096 | 1.37e+03 | 3056 | 3084 | 99% | compute |
| decode batch | batch-1 | 1 | 90 | 91 | 99% | memory |
| decode batch | batch-2 | 2 | 166 | 182 | 91% | memory |
| decode batch | batch-4 | 4 | 324 | 363 | 89% | memory |
| decode batch | batch-8 | 7.98 | 641 | 726 | 88% | memory |
| decode batch | batch-16 | 15.9 | 1309 | 1449 | 90% | memory |
| decode batch | batch-32 | 31.8 | 1477 | 2886 | 51% | memory |
| decode batch | batch-64 | 63 | 3012 | 3084 | 98% | compute |
| decode batch | batch-128 | 124 | 3015 | 3084 | 98% | compute |
| decode batch | batch-256 | 241 | 3010 | 3084 | 98% | compute |
| decode batch | batch-512 | 455 | 3024 | 3084 | 98% | compute |
| decode batch | batch-1024 | 819 | 3010 | 3084 | 98% | compute |

#### Predictions

| Prediction | Result |
|---|---|
| Triad bandwidth between 60 and 100 GB/s | held |
| fp32 n=4096 between 1.0 and 4.0 TFLOPS; fp16 at least 0.9 of it; n=4096 at least 1.2 times n=512 | held |
| Decode sweep: never above its roofline; memory-bound and compute-bound regions reach their floors | held |

#### Machine speed before and after each phase

| Phase | Decode probe before, tokens/s | Decode probe after, tokens/s |
|---|---|---|
| bandwidth | 154.7 | 157.3 |
| matmul | 156.9 | 151.9 |
| skinny | 154.8 | 158.4 |
