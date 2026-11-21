#!/bin/bash
# Download one file from a public Hugging Face model repo and verify its sha256 against the digest
# the repo publishes. Refuses to keep a file whose checksum does not match.
# Usage: scripts/fetch_model.sh <owner/repo> <file>
set -euo pipefail

REPO="$1"
FILE="$2"
DEST="/Users/musa/inference-infra-lab/models"
mkdir -p "$DEST"

expected="$(curl -fsS --max-time 30 "https://huggingface.co/api/models/$REPO?blobs=true" | python3 -c '
import sys, json
want = sys.argv[1]
for s in json.load(sys.stdin).get("siblings", []):
    if s["rfilename"] == want:
        print((s.get("lfs") or {}).get("sha256", ""))
' "$FILE")"
if [ -z "$expected" ]; then echo "no published sha256 for $FILE in $REPO" >&2; exit 3; fi

curl -fL --retry 3 -C - -o "$DEST/$FILE" "https://huggingface.co/$REPO/resolve/main/$FILE"

actual="$(shasum -a 256 "$DEST/$FILE" | cut -d' ' -f1)"
if [ "$actual" != "$expected" ]; then
  echo "CHECKSUM MISMATCH for $FILE" >&2
  echo "  expected $expected" >&2
  echo "  actual   $actual" >&2
  exit 4
fi
echo "$actual  $FILE" > "$DEST/$FILE.sha256"
echo "verified $FILE sha256 $actual ($(du -h "$DEST/$FILE" | cut -f1))"
