#!/bin/bash
# Build llama.cpp from source with Metal, and record exactly what was built.
# Usage: scripts/build_llama.sh [git-ref]     (default: the current default-branch head)
set -euo pipefail

ROOT="/Users/musa/inference-infra-lab"
SRC="$ROOT/third_party/llama.cpp"

if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$SRC"
fi
if [ "${1:-}" != "" ]; then
  git -C "$SRC" fetch --depth 1 origin "$1"
  git -C "$SRC" checkout --quiet FETCH_HEAD
fi

cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build "$SRC/build" --config Release -j 8 --target llama-server llama-bench

echo "commit: $(git -C "$SRC" rev-parse HEAD)"
echo "date:   $(git -C "$SRC" log -1 --format=%cd --date=short)"
"$SRC/build/bin/llama-server" --version 2>&1 | head -3 || true
