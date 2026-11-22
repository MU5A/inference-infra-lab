#!/bin/bash
# Diagnostic only, not a result: does decode speed recover after the machine rests?
cd ~/inference-infra-lab
for wait in 240 300 420; do
  sleep $wait
  printf '%s after +%ss idle: ' "$(date +%H:%M:%S)" "$wait"
  third_party/llama.cpp/build/bin/llama-bench -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf -p 0 -n 64 -r 2 -ngl 99 -o json 2>/dev/null \
    | python3 -c "import json,sys; r=json.load(sys.stdin)[0]; print('tg64', [round(x,1) for x in r['samples_ts']])"
done
