Settings: 5 measured runs after 2 warm-up runs per prompt, max_tokens 128, temperature 0.0, seed 1, prompt cache off. Load average at the start: 3.61. Now drawing from 'AC Power'.

| Prompt | Prompt tokens | TTFT median ms (min to max) | Decode tok/s median (min to max) | ITL ms p50 / p95 / p99 | Reply tokens |
|---|---|---|---|---|---|
| short | 40 | 26.0 (25.8 to 26.7) | 152.6 (143.6 to 156.4) | 6.54 / 7.53 / 8.33 | 56 (stop) |
| medium | 257 | 92.9 (92.8 to 96.9) | 153.2 (147.9 to 155.3) | 6.49 / 7.25 / 8.13 | 128 (length) |
| long | 1469 | 490.8 (489.5 to 496.7) | 149.4 (141.4 to 150.0) | 6.65 / 7.45 / 8.14 | 83 (stop) |

Harness against the server's own timer:

| Prompt | Harness decode tok/s | Server-reported decode tok/s | Difference |
|---|---|---|---|
| short | 152.6 | 152.7 | 0.07% |
| medium | 153.2 | 153.1 | 0.01% |
| long | 149.4 | 149.5 | 0.04% |

Prompt processing rate:

| Prompt | Prompt tokens | Server-reported prefill tok/s | Implied by harness TTFT |
|---|---|---|---|
| short | 40 | 1592 | 1540 |
| medium | 257 | 2805 | 2766 |
| long | 1469 | 3005 | 2993 |
