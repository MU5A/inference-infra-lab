"""Check that the tests can fail: break the code on purpose, one plausible mistake at a time.

Each mutation edits one source file, runs the test suite, and requires it to FAIL. A mutation that
the suite does not catch is a gap in the tests. Files are restored after each mutation, and the
script refuses to start if the source tree has uncommitted changes, so `git checkout -- src`
always recovers the original if a run is interrupted.

Usage:  python scripts/mutation_check.py [substring of a mutation name ...]
Exit status is 1 if any mutation was missed.
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
S = ROOT / "src" / "inferlab"
C = ROOT / "microbench"

MUTATIONS = [
    (
        "percentile: no interpolation",
        S / "stats.py",
        "return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))",
        "return float(ordered[low])",
    ),
    (
        "client: any event counts as text (TTFT too early)",
        S / "client.py",
        "                if text:\n                    result.text_arrivals_ns.append(arrived)",
        "                if True:\n                    result.text_arrivals_ns.append(arrived)",
    ),
    (
        "decode rate off by one",
        S / "client.py",
        "decode_tps = (len(arrivals) - 1) / span",
        "decode_tps = len(arrivals) / span",
    ),
    (
        "usage rate off by one",
        S / "client.py",
        "decode_tps_usage = (result.n_tokens - 1) / span",
        "decode_tps_usage = result.n_tokens / span",
    ),
    (
        "network errors raise instead of being returned",
        S / "client.py",
        "    except httpx.HTTPError as exc:",
        "    except ZeroDivisionError as exc:",
    ),
    (
        "[DONE] treated as malformed",
        S / "client.py",
        '    if payload == "[DONE]":\n        return "done", None\n',
        "",
    ),
    (
        "ITL gaps wrong pairing",
        S / "client.py",
        "gaps = [(b - a) / 1e9 for a, b in zip(arrivals, arrivals[1:], strict=False)]",
        "gaps = [(b - a) / 1e9 for a, b in zip(arrivals, arrivals[2:], strict=False)]",
    ),
    (
        "warm-up requests are measured too",
        S / "runner.py",
        "measured = [metrics(send(payload)) for _ in range(runs)]",
        "measured = [metrics(send(payload)) for _ in range(runs + warmup)]",
    ),
    (
        "cold flag ignored",
        S / "runner.py",
        '    if cold:\n        payload["cache_prompt"] = False',
        '    if False:\n        payload["cache_prompt"] = False',
    ),
    ("failures not counted", S / "runner.py", '"failed": len(runs) - len(ok),', '"failed": 0,'),
    (
        "ITL pooled from first run only",
        S / "runner.py",
        'pooled_itl = [gap for r in ok for gap in r["itl_s"]]',
        'pooled_itl = list(ok[0]["itl_s"]) if ok else []',
    ),
    ("checksum sidecar ignored", S / "env.py", "        if sidecar.exists():", "        if False:"),
    (
        "preflight: load threshold ignored",
        S / "env.py",
        "if load is not None and load > cores * max_load_per_core:",
        "if False:",
    ),
    (
        "preflight: battery never refused",
        S / "env.py",
        'if require_ac and power and "Battery Power" in power:',
        "if False:",
    ),
    (
        "preflight: throttle limit comparison reversed",
        S / "env.py",
        "if int(value) < 100:",
        "if int(value) > 100:",
    ),
    (
        "preflight: override not recorded (CLI)",
        S / "__main__.py",
        "if problems and not args.allow_noisy:",
        "if False:",
    ),
    (
        "gpu: a busy GPU is never refused",
        S / "env.py",
        'if gpu and (gpu["peak"] > max_gpu_peak or gpu["mean"] > max_gpu_mean):',
        "if False:",
    ),
    (
        "gpu: the mean limit is ignored",
        S / "env.py",
        ' or gpu["mean"] > max_gpu_mean',
        "",
    ),
    (
        "stability: the decode limit is disabled",
        S / "runner.py",
        "MAX_DECODE_CV = 0.05",
        "MAX_DECODE_CV = 1e9",
    ),
    (
        "stability: the stable flag is always true",
        S / "runner.py",
        '        "stable": stable,',
        '        "stable": True,',
    ),
    (
        "cli: unstable results exit successfully",
        S / "__main__.py",
        'if not report["stable"]:',
        "if False:",
    ),
    (
        "stats: sample instead of population deviation",
        S / "stats.py",
        "variance = sum((v - mean) ** 2 for v in data) / len(data)",
        "variance = sum((v - mean) ** 2 for v in data) / (len(data) - 1)",
    ),
    (
        "chase: Fisher-Yates instead of Sattolo (short cycles allowed)",
        C / "chase.c",
        "rng_next(&state) % i);",
        "rng_next(&state) % (i + 1));",
    ),
    (
        "chase: every node points at itself",
        C / "chase.c",
        "base + (size_t)next[i] * node_bytes;",
        "base + i * node_bytes;",
    ),
    (
        "membw: the triad kernel computes the wrong thing",
        C / "membw.c",
        "a[i] = b[i] + SCALAR * c[i];",
        "a[i] = b[i] + c[i];",
    ),
    (
        "membw: the scale kernel does nothing",
        C / "membw.c",
        "for (size_t i = j->lo; i < j->hi; i++) b[i] = SCALAR * c[i];",
        "(void)b;",
    ),
    (
        "matmul: ikj forgets to zero C between calls",
        C / "matmul.c",
        "static void mm_ikj(int n, const float *A, const float *B, float *C) {\n    memset(C, 0, (size_t)n * n * sizeof(float));",
        "static void mm_ikj(int n, const float *A, const float *B, float *C) {",
    ),
    (
        "matmul: blocked tiles run past the edge",
        C / "matmul.c",
        "jmax = jj + bs < n ? jj + bs : n;",
        "jmax = jj + bs;",
    ),
    (
        "mmap_touch: the stride is off by one",
        C / "mmap_touch.c",
        "off += stride) checksum",
        "off += stride + 1) checksum",
    ),
    (
        "copy_nocache: an existing destination is appended to",
        C / "copy_nocache.c",
        "O_WRONLY | O_CREAT | O_TRUNC",
        "O_WRONLY | O_CREAT | O_APPEND",
    ),
    (
        "memory: plateau tolerance is far too tight",
        S / "memory.py",
        "PLATEAU_TOLERANCE = 0.30",
        "PLATEAU_TOLERANCE = 0.01",
    ),
    (
        "memory: two points are enough for a plateau",
        S / "memory.py",
        "MIN_PLATEAU_POINTS = 3",
        "MIN_PLATEAU_POINTS = 2",
    ),
    (
        "memory: the bandwidth stability limit is loosened",
        S / "memory.py",
        '"bandwidth": 0.10',
        '"bandwidth": 0.90',
    ),
    (
        "memory: low free memory is never refused",
        S / "env.py",
        "if free is not None and free < min_free_memory_percent:",
        "if False:",
    ),
    (
        "memory: the free-percentage regex reads the wrong number",
        S / "env.py",
        'r"free percentage:\\s*(\\d+)%"',
        'r"(\\d+)%"',
    ),
    (
        "env: busiest processes sorted the wrong way round",
        S / "env.py",
        "rows.sort(key=lambda r: r[2], reverse=True)",
        "rows.sort(key=lambda r: r[2])",
    ),
    (
        "env: the benchmark's own process tree is not excluded",
        S / "env.py",
        "rows = [r for r in rows if r[0] not in excluded]",
        "rows = list(rows)",
    ),
    (
        "env: executable path is kept instead of just the name",
        S / "env.py",
        'return [{"cpu_percent": r[2], "name": Path(r[3]).name} for r in rows[:n]]',
        'return [{"cpu_percent": r[2], "name": r[3]} for r in rows[:n]]',
    ),
    (
        "memory: a cold read need only match a cold touch",
        S / "memory.py",
        'if trial["cold_read"]["gbs"] < 5 * plain:',
        'if trial["cold_read"]["gbs"] < 0 * plain:',
    ),
    (
        "memory: the willneed prediction is never checked",
        S / "memory.py",
        'if trial["cold_willneed"]["gbs"] < 3 * plain:',
        "if False:",
    ),
    (
        "matmul: every sample times a single call, however short",
        C / "matmul.c",
        "int calls = probe_ns >= MIN_SAMPLE_NS ? 1 :",
        "int calls = probe_ns >= 0 ? 1 :",
    ),
    (
        "crosscheck: a machine running far below the baseline is accepted",
        S / "crosscheck.py",
        "if measured_tps < floor:",
        "if False:",
    ),
    (
        "crosscheck: the reference floor is exclusive instead of inclusive",
        S / "crosscheck.py",
        "if measured_tps < floor:",
        "if measured_tps <= floor:",
    ),
    (
        "crosscheck: a missing reference probe is accepted",
        S / "crosscheck.py",
        "if measured_tps is None:",
        "if False:",
    ),
    (
        "crosscheck: the cache speed-up prediction is never checked",
        S / "crosscheck.py",
        'if row["speedup"] < min_speedup:',
        "if False:",
    ),
    (
        "crosscheck: a cold run that reused cached tokens passes",
        S / "crosscheck.py",
        'if row["cold_cached_tokens"] != 0:',
        "if False:",
    ),
    (
        "crosscheck: llama-bench variation is not limited",
        S / "crosscheck.py",
        "MAX_BENCH_CV = 0.05",
        "MAX_BENCH_CV = 9.0",
    ),
    (
        "crosscheck: the decode tolerance is loosened",
        S / "crosscheck.py",
        "DECODE_TOLERANCE = 0.10",
        "DECODE_TOLERANCE = 0.90",
    ),
    (
        "overhead: the upper bound takes the best run, not the worst",
        S / "overhead.py",
        'return {k: max(r["overhead_us"][k] for r in runs) for k in keys}',
        'return {k: min(r["overhead_us"][k] for r in runs) for k in keys}',
    ),
    (
        "watch: contended is never reported",
        S / "watch.py",
        '"contended": peak is not None and peak > CONTENDED_CPU_PERCENT,',
        '"contended": False,',
    ),
    (
        "watch: wait_quiet never waits",
        S / "watch.py",
        "if not problems or clock() >= deadline:",
        "if True:",
    ),
    (
        "usage not captured",
        S / "client.py",
        '                if isinstance(event.get("usage"), dict):\n                    result.usage = event["usage"]\n',
        "",
    ),
    (
        "roofline: the ridge is bandwidth over compute",
        S / "roofline.py",
        "return self.peak_flops / self.bandwidth",
        "return self.bandwidth / self.peak_flops",
    ),
    (
        "roofline: attainable ignores the compute ceiling",
        S / "roofline.py",
        "return min(self.peak_flops, self.bandwidth * intensity)",
        "return self.bandwidth * intensity",
    ),
    (
        "roofline: the ridge itself counts as memory-bound",
        S / "roofline.py",
        'return "memory" if intensity < self.ridge else "compute"',
        'return "memory" if intensity <= self.ridge else "compute"',
    ),
    (
        "roofline: a product counts one operation per multiply-add",
        S / "roofline.py",
        "return 2.0 * m * n * k",
        "return 1.0 * m * n * k",
    ),
    (
        "roofline: output traffic is left out of the product's bytes",
        S / "roofline.py",
        "return (m * k + k * n) * in_bytes + m * n * out",
        "return (m * k + k * n) * in_bytes",
    ),
    (
        "roofline: the knee ignores activation traffic",
        S / "roofline.py",
        "denominator = 2.0 * k * n - ridge * (k * act_bytes + n * out_bytes)",
        "denominator = 2.0 * k * n",
    ),
    (
        "roofline: an unreachable knee divides by zero",
        S / "roofline.py",
        "if denominator <= 0:",
        "if denominator < 0:",
    ),
    (
        "roofline: the 5% tolerance above the roof is dropped",
        S / "roofline.py",
        '"above_roof": fraction > 1.0 + ROOF_TOLERANCE,',
        '"above_roof": fraction > 1.0,',
    ),
    (
        "roofline: rates ignore the time taken",
        S / "roofline.py",
        'flop_rates = [r[key]["flops"] / r[key]["best_s"] for r in runs]',
        'flop_rates = [r[key]["flops"] for r in runs]',
    ),
    (
        "roofline: the prediction range is exclusive",
        S / "roofline.py",
        "return low <= value <= high",
        "return low < value < high",
    ),
    (
        "roofline: half precision may be much slower than single",
        S / "roofline.py",
        "if fp16_large_tflops < FP16_VS_FP32_MIN_RATIO * fp32_large_tflops:",
        "if False:",
    ),
    (
        "roofline: the large size need not beat the small one",
        S / "roofline.py",
        "if fp32_large_tflops < LARGE_VS_SMALL_MIN_RATIO * fp32_small_tflops:",
        "if False:",
    ),
    (
        "roofline: a point above its roofline is not reported",
        S / "roofline.py",
        'if placed["above_roof"]:',
        "if False:",
    ),
    (
        "roofline: the memory-bound region edge is exclusive",
        S / "roofline.py",
        'if p["batch"] <= MEMORY_REGION_KNEE_FRACTION * knee:',
        'if p["batch"] < MEMORY_REGION_KNEE_FRACTION * knee:',
    ),
    (
        "roofline: the compute-bound region edge is exclusive",
        S / "roofline.py",
        'if p["batch"] >= COMPUTE_REGION_KNEE_MULTIPLE * knee:',
        'if p["batch"] > COMPUTE_REGION_KNEE_MULTIPLE * knee:',
    ),
    (
        "roofline: a figure need not name its source",
        S / "roofline.py",
        'if not str(fig.get("source", "")).startswith("https://"):',
        "if False:",
    ),
    (
        "roofline: a derived figure need not give its rule",
        S / "roofline.py",
        'if fig.get("how") == "derived" and not (fig.get("from") and fig.get("rule")):',
        "if False:",
    ),
    (
        "roofline: terabytes per second are scaled as gigabytes",
        S / "roofline.py",
        '"TB/s": 1e12,',
        '"TB/s": 1e9,',
    ),
    (
        "report: an unstable result may set a roof",
        S / "roofline_report.py",
        'return bool(entry) and entry["stable"] and entry["validated"]',
        'return bool(entry) and entry["validated"]',
    ),
    (
        "report: a wrong result may set a roof",
        S / "roofline_report.py",
        'return bool(entry) and entry["stable"] and entry["validated"]',
        'return bool(entry) and entry["stable"]',
    ),
    (
        "report: the bandwidth roof is the worst kernel",
        S / "roofline_report.py",
        "return max(values) if values else None",
        "return min(values) if values else None",
    ),
    (
        "report: single-precision points use the half-precision roof",
        S / "roofline_report.py",
        'roof = roof32 if series == "matmul fp32" else roof16',
        "roof = roof16",
    ),
    (
        "report: the decode knee assumes four-byte weights",
        S / "roofline_report.py",
        "knee = rl.knee_batch(k, n, 2, 2, 2, roof16.ridge)",
        "knee = rl.knee_batch(k, n, 4, 2, 2, roof16.ridge)",
    ),
    (
        "report: an unevaluated prediction is shown as held",
        S / "roofline_report.py",
        'result = "not evaluated: its inputs were not citable"',
        'result = "held"',
    ),
    (
        "report: derived specification figures are not marked",
        S / "roofline_report.py",
        'mark = " (derived)" if fig["how"] == "derived" else ""',
        'mark = ""',
    ),
    (
        "report: the CPU roof may use the background-priority triad",
        S / "roofline_report.py",
        'if key.startswith("interactive-") and c["kernels"]["triad"]["stable"]',
        'if c["kernels"]["triad"]["stable"]',
    ),
    (
        "chart: the axes are linear, not logarithmic",
        S / "roofline_report.py",
        "fraction = (math.log10(value) - math.log10(low)) / (math.log10(high) - math.log10(low))",
        "fraction = (value - low) / (high - low)",
    ),
    (
        "chart: a roof turns flat at the wrong intensity",
        S / "roofline_report.py",
        "(_px(ridge), _py(peak)),",
        "(_px(ridge * 2), _py(peak)),",
    ),
    (
        "chart: names are not escaped",
        S / "roofline_report.py",
        '{escape(roof["name"])}</text>\')',
        '{roof["name"]}</text>\')',
    ),
    (
        "chart: the axis floor hides the lowest streaming point",
        S / "roofline_report.py",
        "Y_RANGE = (1e9, 1e16)",
        "Y_RANGE = (1e10, 1e16)",
    ),
    (
        "chart: the legend margin is too narrow for its labels",
        S / "roofline_report.py",
        "LEFT, RIGHT, TOP, BOTTOM = 100, 280, 24, 58",
        "LEFT, RIGHT, TOP, BOTTOM = 100, 150, 24, 58",
    ),
    (
        "metal: triad traffic counts only two arrays",
        S / "metal_sweep.py",
        "return 12 * elements",
        "return 8 * elements",
    ),
    (
        "metal: an FMA counted as one operation",
        S / "metal_sweep.py",
        "return 2.0 * threads * chains * iterations",
        "return 1.0 * threads * chains * iterations",
    ),
    (
        "metal: dispatch check ignores what ran",
        S / "metal_sweep.py",
        "seen == {asked},",
        "True,",
    ),
    (
        "metal: stream floor against MLX ignored",
        S / "metal_sweep.py",
        "    if best < floor:",
        "    if best < 0:",
    ),
    (
        "metal: threadgroup 32 judged for flatness",
        S / "metal_sweep.py",
        "if tg < STREAM_FLAT_FROM or key not in rates:",
        "if key not in rates:",
    ),
    (
        "metal: flatness tolerance doubled",
        S / "metal_sweep.py",
        "if rates[key] < (1 - STREAM_FLAT_TOLERANCE) * top:",
        "if rates[key] < (1 - 2 * STREAM_FLAT_TOLERANCE) * top:",
    ),
    (
        "metal: occupancy drop measured from the wrong end",
        S / "metal_sweep.py",
        "drop = 1 - rates[-1] / rates[0]",
        "drop = 1 - rates[0] / rates[-1]",
    ),
    (
        "metal: occupancy rise check removed",
        S / "metal_sweep.py",
        "if high > (1 + OCCUPANCY_MAX_RISE) * low:",
        "if False:",
    ),
    (
        "metal: best FMA is the worst threadgroup",
        S / "metal_sweep.py",
        'return max(_median(summary, fma_key(dtype, tg), "flops_per_s") for tg in THREADGROUP_SIZES)',
        'return min(_median(summary, fma_key(dtype, tg), "flops_per_s") for tg in THREADGROUP_SIZES)',
    ),
    (
        "metal: FMA against matmul check removed",
        S / "metal_sweep.py",
        "if fp32 < FP32_FMA_MIN_FRACTION_OF_MATMUL * mlx_fp32_matmul_flops:",
        "if False:",
    ),
    (
        "metal: fp16 ratio inverted",
        S / "metal_sweep.py",
        "ratio = fp16 / fp32",
        "ratio = fp32 / fp16",
    ),
    (
        "metal: fp32 FMA range check removed",
        S / "metal_sweep.py",
        "if not low <= fp32 / 1e12 <= high:",
        "if False:",
    ),
    (
        "metal: a failed prediction reads as held",
        S / "metal_sweep.py",
        '"held": not failures[phase],',
        '"held": True,',
    ),
    (
        "metal report: dispatch ignored for citability",
        S / "metal_report.py",
        'and entry["validated"] and entry["dispatched_as_asked"]',
        'and entry["validated"]',
    ),
    (
        "metal report: share uses the wrong reference",
        S / "metal_report.py",
        'share = entry["bytes_per_s"]["median"] / mlx_triad if citable(entry) else None',
        'share = entry["bytes_per_s"]["median"] / 80e9 if citable(entry) else None',
    ),
    (
        "metal report: occupancy relative to the largest size",
        S / "metal_report.py",
        "base = s.get(ms.occupancy_key(ms.SCRATCH_BYTES[0]))",
        "base = s.get(ms.occupancy_key(ms.SCRATCH_BYTES[-1]))",
    ),
    (
        "metal report: a phase with a non-citable configuration is judged",
        S / "metal_report.py",
        "        if not all(citable(e) for e in summary.values()):",
        "        if False:",
    ),
    (
        "kvmem: grouped-query cache counts every attention head",
        S / "kvmem.py",
        "return m.kv_heads * m.head_dim, m.kv_heads * m.v_head_dim",
        "return m.heads * m.head_dim, m.heads * m.v_head_dim",
    ),
    (
        "kvmem: latent attention cache forgets the rotary part",
        S / "kvmem.py",
        "return m.kv_lora_rank + m.qk_rope, 0",
        "return m.kv_lora_rank, 0",
    ),
    (
        "kvmem: expanded latent cache drops the values",
        S / "kvmem.py",
        "return m.heads * (m.qk_nope + m.qk_rope), m.heads * m.v_head_dim",
        "return m.heads * (m.qk_nope + m.qk_rope), 0",
    ),
    (
        "kvmem: q8_0 block size wrong",
        S / "kvmem.py",
        '"q8_0": (34, 32),',
        '"q8_0": (32, 32),',
    ),
    (
        "kvmem: q4_0 block size wrong",
        S / "kvmem.py",
        '"q4_0": (18, 32),',
        '"q4_0": (16, 32),',
    ),
    (
        "kvmem: a quantized row that is not whole blocks is accepted",
        S / "kvmem.py",
        "    if elements % block_elements:",
        "    if False:",
    ),
    (
        "kvmem: tied embeddings counted twice",
        S / "kvmem.py",
        "head = 0 if m.tied else embedding",
        "head = embedding",
    ),
    (
        "kvmem: active parameters count every expert",
        S / "kvmem.py",
        "return m.experts * expert + shared + router, m.experts_per_token * expert + shared + router",
        "return m.experts * expert + shared + router, m.experts * expert + shared + router",
    ),
    (
        "kvmem: shared experts left out of the active count",
        S / "kvmem.py",
        "return m.experts * expert + shared + router, m.experts_per_token * expert + shared + router",
        "return m.experts * expert + shared + router, m.experts_per_token * expert + router",
    ),
    (
        "kvmem: leading dense layers treated as sparse",
        S / "kvmem.py",
        "            and index >= self.first_dense_layers",
        "            and index >= 0",
    ),
    (
        "kvmem: sparse layer frequency ignored",
        S / "kvmem.py",
        "            and index % self.moe_layer_freq == 0",
        "            and True",
    ),
    (
        "kvmem: attention bias left out",
        S / "kvmem.py",
        "    if m.qkv_bias:\n        total += q_out + k_out + v_out",
        "    if False:\n        total += q_out + k_out + v_out",
    ),
    (
        "kvmem: query and key norms left out",
        S / "kvmem.py",
        "    if m.qk_norm:\n        total += q_out + k_out",
        "    if False:\n        total += q_out + k_out",
    ),
    (
        "kvmem: weight bytes rounded down",
        S / "kvmem.py",
        "return math.ceil(parameters * bits_per_weight / 8)",
        "return int(parameters * bits_per_weight / 8)",
    ),
    (
        "kvmem: maximum context one token too many",
        S / "kvmem.py",
        "    return max(0, room // per_token)",
        "    return max(0, room // per_token + 1)",
    ),
    (
        "kvmem: a mixtral sliding window is accepted",
        S / "kvmem.py",
        'model_type in ("mistral", "mixtral") and sliding is not None',
        'model_type == "mistral" and sliding is not None',
    ),
    (
        "kvmem: an active sliding window is accepted",
        S / "kvmem.py",
        'if cfg.get("use_sliding_window") is True or (',
        "if False or (",
    ),
    (
        "kvmem: a missing tie flag is accepted",
        S / "kvmem.py",
        '    if "tie_word_embeddings" not in cfg:',
        "    if False:",
    ),
    (
        "kvmem: the separate output matrix option does nothing",
        S / "kvmem.py",
        "if args.output_matrix_separate and model.tied:",
        "if False and model.tied:",
    ),
    (
        "gguf: the counts in the header are not checked against the file",
        S / "gguf.py",
        "if n_metadata * 12 > r.remaining() or n_tensors * 24 > r.remaining():",
        "if False:",
    ),
    (
        "gguf: tensor data offset not aligned",
        S / "gguf.py",
        "data_offset = _align(handle.tell(), alignment)",
        "data_offset = handle.tell()",
    ),
    (
        "gguf: a tensor past the end of the file is accepted",
        S / "gguf.py",
        "        if data_offset + t.offset + t.nbytes > size:",
        "        if False:",
    ),
    (
        "gguf: Q4_K block size wrong",
        S / "gguf.py",
        '12: ("Q4_K", 256, 144),',
        '12: ("Q4_K", 256, 140),',
    ),
    (
        "gguf: padded size ignores alignment",
        S / "gguf.py",
        "return sum(_align(t.nbytes, self.alignment) for t in self.tensors)",
        "return sum(t.nbytes for t in self.tensors)",
    ),
    (
        "gguf: a tied model still expected to have an output matrix",
        S / "gguf.py",
        "    if not m.tied:",
        "    if True:",
    ),
    (
        "gguf: an edited summary is not validated",
        S / "gguf.py",
        "    _check_tensors(infos, alignment, data_offset, file_size)",
        "    pass",
    ),
    (
        "kvcheck: KV tolerance loosened to 5%",
        S / "kvcheck.py",
        "KV_TOLERANCE = 0.005",
        "KV_TOLERANCE = 0.05",
    ),
    (
        "kvcheck: one launch is enough",
        S / "kvcheck.py",
        "LAUNCHES_REQUIRED = 3",
        "LAUNCHES_REQUIRED = 1",
    ),
    (
        "kvcheck: the published rounding allowance doubled",
        S / "kvcheck.py",
        "<= 0.5 * 10**-decimals + 1e-9",
        "<= 1.0 * 10**-decimals + 1e-9",
    ),
    (
        "kvcheck: the context column allowance widened",
        S / "kvcheck.py",
        "CONTEXT_COLUMN_MIB = 2",
        "CONTEXT_COLUMN_MIB = 20",
    ),
    (
        "kvcheck: any ratio passes the latent attention check",
        S / "kvcheck.py",
        "MLA_RATIO_RANGE = (8.0, 10.0)",
        "MLA_RATIO_RANGE = (1.0, 100.0)",
    ),
    (
        "kvcheck: padding need not exceed the asked size",
        S / "kvcheck.py",
        "        if reported < (1 + PADDING_MIN_OVER) * at_asked:",
        "        if False:",
    ),
    (
        "kvcheck: launches may disagree on the KV size",
        S / "kvcheck.py",
        '        for field in ("cells", "kv_mib"):',
        '        for field in ("cells",):',
    ),
    (
        "kvcheck: a launch with no breakdown is usable",
        S / "kvcheck.py",
        '    if parsed["breakdown"] is None:',
        "    if False:",
    ),
    (
        "kvcheck: unusable launches are summarized",
        S / "kvcheck.py",
        "        if launch_problems(launch):\n            continue",
        "        if False:\n            continue",
    ),
    (
        "llama_log: the first memory breakdown is used, not the last",
        S / "llama_log.py",
        "breakdowns[-1]",
        "breakdowns[0]",
    ),
    (
        "llama_log: billions read as millions",
        S / "llama_log.py",
        '"B": 1e3,',
        '"B": 1.0,',
    ),
    (
        "llama_log: flash attention in auto mode counts as on",
        S / "llama_log.py",
        'm[1] in ("enabled", "on")',
        'm[1] in ("enabled", "on", "auto")',
    ),
    (
        "llama_log: the resolved flash attention line is ignored",
        S / "llama_log.py",
        "        elif FLASH_ON.search(msg):\n            flash = True",
        "        elif FLASH_ON.search(msg):\n            flash = False",
    ),
    (
        "llama_log: breakdown columns are not added",
        S / "llama_log.py",
        "if isinstance(r.get(column), int))",
        "if False)",
    ),
    (
        "kvreport: the printed parameter count is used when launches disagree",
        S / "kvreport.py",
        "return seen.pop() if len(seen) == 1 else None",
        "return seen.pop() if seen else None",
    ),
    (
        "kvreport: the scenario weights use 8 bits",
        S / "kvreport.py",
        "SCENARIO_BITS = 16.0",
        "SCENARIO_BITS = 8.0",
    ),
    (
        "kvreport: the K-quant observation uses the wrong divisor",
        S / "kvreport.py",
        "    divisible = [t for t in weights if t.dims[0] % 256 == 0]",
        "    divisible = [t for t in weights if t.dims[0] % 128 == 0]",
    ),
]


def main() -> int:
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--", "src"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        print("refusing to run: src/ has uncommitted changes\n" + dirty, file=sys.stderr)
        return 2
    only = sys.argv[1:]
    missed = 0
    for name, path, old, new in MUTATIONS:
        if only and not any(o in name for o in only):
            continue
        original = path.read_text()
        if old not in original:
            print(f"SKIP (pattern not found) {name}")
            missed += 1
            continue
        path.write_text(original.replace(old, new, 1))
        try:
            run = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
        finally:
            path.write_text(original)
        caught = run.returncode != 0
        failed = next((ln for ln in run.stdout.splitlines() if ln.startswith("FAILED")), "")
        print(f"{'CAUGHT ' if caught else 'MISSED!'} {name}")
        if caught and failed:
            print(f"          {failed[:110]}")
        missed += not caught
    print(f"missed or skipped: {missed}")
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
