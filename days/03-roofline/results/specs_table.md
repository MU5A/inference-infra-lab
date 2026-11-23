| Device | Memory bandwidth | FP16/BF16 tensor, dense | FP32 | Ridge, FLOP/byte | Decode batch to reach the ridge (fp16 weights) |
|---|---|---|---|---|---|
| NVIDIA A100 80GB SXM | 2039 GB/s | 312 TFLOPS | 19.5 TFLOPS | 153 | 159 |
| NVIDIA H100 SXM | 3.35 TB/s | 989.5 TFLOPS (derived) | 67 TFLOPS | 295 | 318 |
| NVIDIA H200 SXM | 4.8 TB/s | 989.5 TFLOPS (derived) | 67 TFLOPS | 206 | 217 |
| NVIDIA B200 SXM (per GPU) | up to 8 TB/s | 2250 TFLOPS (derived) | not published by the vendor | 281 | 302 |
| AMD Instinct MI300X | 5.3 TB/s | 1307.4 TFLOPS | 163.4 TFLOPS | 247 | 262 |
| Apple M3 (this machine: 10-core GPU, 16 GB) | 100 GB/s | not published by the vendor | not published by the vendor | not published | not published |
