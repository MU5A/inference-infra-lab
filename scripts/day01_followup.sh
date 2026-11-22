#!/bin/bash
# The three Day 1 measurements that need a quiet machine. Every step is gated on the pre-flight
# check and does not run on a busy, throttled or battery-powered machine.
#   step1  cold vs warm prompt cache  (needs the llama-server from the baseline on :8081)
#   step2  the harness's own timing error, three runs, against a fake server
#   step3  llama-bench cross-check    (stops the server first so the two do not share the GPU)
# Usage: scripts/day01_followup.sh [step1] [step2] [step3]     (default: all three)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/days/01-harness-and-baseline/results"
MODEL="$ROOT/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
LLAMA="$ROOT/third_party/llama.cpp"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"

gate() {
  "$PY" -c "import sys; from inferlab import env; p = env.preflight(); print('preflight:', p or 'clean'); sys.exit(1 if p else 0)"
}

step1() {
  gate || return 1
  "$PY" -m inferlab bench --prompts prompts/day01.json --runs 5 --warmup 2 --max-tokens 128 \
    --warm-cache --model "$MODEL" --llama-dir "$LLAMA" --out "$OUT/warm_cache.json"
}

step2() {
  gate || return 1
  "$PY" scripts/harness_overhead.py
}

step3() {
  pkill -x llama-server && sleep 3   # exact process name only, never a command-line match
  gate || return 1
  "$LLAMA/build/bin/llama-bench" -m "$MODEL" -p 512 -n 128 -r 5 -ngl 99 -o json 2>/dev/null > "$OUT/llama_bench.json" || return 1
  "$PY" -c "
import json
from inferlab import env
print(json.dumps({'preflight_problems': env.preflight(), 'environment': env.describe()}, indent=2))
" > "$OUT/llama_bench.environment.json"
}

steps=("$@"); [ ${#steps[@]} -eq 0 ] && steps=(step1 step2 step3)
for s in "${steps[@]}"; do
  echo "== $s"
  "$s" || echo "$s did not run or failed"
done
