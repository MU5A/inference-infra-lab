| CUDA concept | What the CUDA guide says (L, fetched 2026-11-24) | Nearest Metal concept | What today shows on this chip |
|---|---|---|---|
| Warp | Threads in a block are grouped 32 at a time into warps that run in SIMT fashion [1]; `warpSize` is a run-time value, commonly 32 [3] | SIMD group (`threads_per_simdgroup`) | M: every kernel in every run reported a SIMD-group width of 32 (dispatch table) |
| Thread block | Threads that can share memory and synchronize; at most 1024 threads in the compute capability 10.0 example [2] | Threadgroup | M: every threadgroup size from 32 to 1024 compiled, validated and ran as asked for all three kernels (dispatch table). Larger sizes were not tried |
| Streaming Multiprocessor (SM) | The GPU is a collection of SMs, each with registers, a unified data cache and functional units [1] | GPU core | Not measured. `system_profiler` reports 10 GPU cores for this chip |
| Shared memory | Per-block scratchpad on the SM, sharing physical storage with the L1 cache [1][2] | Threadgroup memory (`threadgroup` arrays) | M: arrays from 1 KiB to 32 KiB per threadgroup compiled and were used by every thread (occupancy table). The on-chip storage behind them is not documented here |
| `__syncthreads()` | Blocks every thread of a block until all reach it [2] | `threadgroup_barrier(mem_flags::mem_threadgroup)` | M, as correctness only: the occupancy kernel's neighbour exchange validated in every configuration, which needs the barrier |
| Occupancy | Active warps over the most an SM supports; shared memory, registers and threads per block all limit it [2] | Same idea: threadgroup memory, registers and threads per group limit how many groups a core holds | M: the occupancy table measures bandwidth against threadgroup memory declared. The occupancy itself is not read: Metal through MLX exposes no counter for it here |
| Tensor Cores, warp matrix functions | Warp-wide matrix operations of the form D = A * B + C that use Tensor Cores [3] | `simdgroup_matrix` 8 x 8 operations; MLX 0.32.4's matrix-multiply kernels (`steel/gemm/mma.h`) build on them, two elements per thread | L only: read in MLX's installed source. Whether this chip has separate matrix units is not shown by anything run today |

Sources, fetched 2026-11-24 (the old URL `cuda-c-programming-guide` now redirects to these):

1. NVIDIA, CUDA Programming Guide, 1.2 Programming model:
   https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html
2. NVIDIA, CUDA Programming Guide, 2.3 Writing CUDA kernels:
   https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html
3. NVIDIA, CUDA Programming Guide, 5.4 C++ language extensions:
   https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html
