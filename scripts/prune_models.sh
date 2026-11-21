#!/bin/bash
# Delete model files. This is the only deletion the runbook allows, and it refuses any path that
# is not directly inside /Users/musa/inference-infra-lab/models/.
# Usage: scripts/prune_models.sh file.gguf [file2.gguf ...]
set -euo pipefail

MODELS="/Users/musa/inference-infra-lab/models"

for name in "$@"; do
  case "$name" in
    */*|""|.|..) echo "refusing '$name': give a bare file name inside models/" >&2; exit 2 ;;
  esac
  target="$MODELS/$name"
  if [ ! -f "$target" ]; then echo "not found: $target" >&2; exit 2; fi
  case "$target" in
    *.gguf|*.safetensors) ;;
    *) echo "refusing '$name': only .gguf and .safetensors files" >&2; exit 2 ;;
  esac
  rm -- "$target"
  echo "removed $target"
done
