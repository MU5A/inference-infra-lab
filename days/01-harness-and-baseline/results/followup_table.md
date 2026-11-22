Harness overhead, three runs against a fake server with known timing:

| Run | Events | Median us | p95 us | p99 us | Max us |
|---|---|---|---|---|---|
| 1 | 3450 | 268 | 448 | 738 | 5728 |
| 2 | 3450 | 268 | 422 | 741 | 3465 |
| 3 | 3450 | 271 | 447 | 756 | 3016 |
| **Upper bound** (worst of each column) | | 271 | 448 | 756 | 5728 |

Events that appeared to arrive before they were sent: 0. Client CPU per event with the profiler attached: 744 us (the profiler inflates this).

Cold against warm prompt cache, one session:

| Prompt | Prompt tokens | Cold TTFT ms | Warm TTFT ms | Saved ms | Speed-up | Cached tokens reused per warm request | Predicted at least 5x |
|---|---|---|---|---|---|---|---|
| short | 40 | 25.5 | 8.0 | 17.5 | 3.2x | 39 | did not hold |
| medium | 257 | 91.9 | 8.3 | 83.6 | 11.1x | 256 | held |
| long | 1469 | 485.3 | 10.3 | 475.0 | 47.2x | 1468 | held |

Decode speed probes either side of each phase, to show the machine stayed up to speed:

| Phase | Decode probe before, tokens/s | Decode probe after, tokens/s |
|---|---|---|
| cold | 164.6 | 151.6 |
| warm | 154.5 | 160.4 |

llama-bench cross-check:

| Measurement | llama-bench tokens/s (standard deviation) | Variation | Compared with | Difference | Predicted within |
|---|---|---|---|---|---|
| pp512 | 3191.9 (7.4) | 0.2% | server prefill, long prompt 3005.2 | +6.2% | 15%: held |
| tg128 | 159.7 (3.7) | 2.3% | harness decode baseline 152.6 | +4.7% | 10%: held |

Decode speed probes either side of the run:

| Phase | Decode probe before, tokens/s | Decode probe after, tokens/s |
|---|---|---|
| bench | 160.2 | 160.9 |
