#!/bin/bash
# Combine split parts into the full quantized bin file.
# Usage: ./combine.sh [output_dir]
#
# If output_dir is omitted the combined file is written
# into the same directory as this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${1:-$SCRIPT_DIR}"

OUT_FILE="$OUT_DIR/nntr_qwen3_0.6b_w4e6a32.bin"
EXPECTED_SHA256="eed53478ffd0d72241efde66bf505de74b3c63ff12f5bf0e363a0329f8c0ee2a"

echo "Combining parts -> $OUT_FILE"
cat "$SCRIPT_DIR"/nntr_qwen3_0.6b_w4e6a32.bin.part_* > "$OUT_FILE"

ACTUAL=$(sha256sum "$OUT_FILE" | awk '{print $1}')
if [ "$ACTUAL" = "$EXPECTED_SHA256" ]; then
  echo "OK  sha256 verified ($ACTUAL)"
else
  echo "FAIL  expected $EXPECTED_SHA256"
  echo "      got      $ACTUAL"
  exit 1
fi
