#!/bin/bash
# Reproducible recipe to generate Qwen3-0.6B x86 weights with Q4_0 FC and
# Q6_K embedding + LM head. The Q6_K LM head is required for inference:
# Qwen3-0.6B is tied (tie_word_embeddings=true), and the nntrainer tied
# embedding layer currently accepts only Q6_K or FP32 weights on the tied
# path. Q4_0 LM head => "Tieword embedding is not supported yet for the
# data type" at the first decode step.
#
# Requirements:
#   - x86_64 Linux host (Q4_0 is platform-specific)
#   - Python 3 with torch, transformers, numpy, safetensors
#   - Either:
#       * nntrainer built with -Denable-transformer=true, NNTRAINER env var
#         pointing at the checkout root (provides nntr_quantize); OR
#       * QUICK_DOT_AI_QUANTIZE env var pointing at a built
#         Quick.AI quick_dot_ai_quantize binary
#     Both accept --fc_dtype/--embd_dtype/--lmhead_dtype with the same flags.
#
# Usage:
#   export NNTRAINER=/path/to/nntrainer
#   ./convert_qwen3_0.6b_q6k_lmhead.sh
#
# Or with Quick.AI's quantize binary:
#   export QUICK_DOT_AI_QUANTIZE=/path/to/Quick.AI/build/quick_dot_ai_quantize
#   ./convert_qwen3_0.6b_q6k_lmhead.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$REPO_ROOT/qwen3-0.6b-q40-q6k-x86"
SIBLING_BUNDLE="$REPO_ROOT/qwen3-0.6b-q40-x86"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Pick a quantizer. QUICK_DOT_AI_QUANTIZE wins if set (it is self-contained
# and does not need an nntrainer transformer build).
if [ -n "${QUICK_DOT_AI_QUANTIZE:-}" ]; then
  QUANTIZER="$QUICK_DOT_AI_QUANTIZE"
elif [ -n "${NNTRAINER:-}" ]; then
  QUANTIZER="$NNTRAINER/build/Applications/CausalLM/nntr_quantize"
else
  echo "Set either QUICK_DOT_AI_QUANTIZE or NNTRAINER before running." >&2
  exit 1
fi

if [ ! -x "$QUANTIZER" ]; then
  echo "Quantizer not executable: $QUANTIZER" >&2
  exit 1
fi

# weight_converter.py lives under the nntrainer tree. If NNTRAINER is unset
# we still require a path to it.
CONVERTER="${WEIGHT_CONVERTER:-}"
if [ -z "$CONVERTER" ]; then
  if [ -n "${NNTRAINER:-}" ]; then
    CONVERTER="$NNTRAINER/Applications/CausalLM/res/qwen3/qwen3-4b/weight_converter.py"
  else
    echo "Set WEIGHT_CONVERTER or NNTRAINER to point at weight_converter.py" >&2
    exit 1
  fi
fi

if [ ! -f "$CONVERTER" ]; then
  echo "weight_converter.py not found at $CONVERTER" >&2
  exit 1
fi

echo "=== Step 1: Stage Qwen/Qwen3-0.6B HF assets ==="
# tokenizer / config files are already committed alongside the sibling
# Q4_0 bundle; reuse them rather than re-downloading. This avoids a hard
# dependency on huggingface.co for the small JSON/text files and only the
# safetensors weights file actually needs the network.
mkdir -p "$WORK/hf"
HF_BASE="https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main"

stage_file() {
  local f="$1"
  if [ -f "$SIBLING_BUNDLE/$f" ]; then
    echo "  reusing $f from $SIBLING_BUNDLE"
    cp "$SIBLING_BUNDLE/$f" "$WORK/hf/$f"
  else
    echo "  fetching $f from $HF_BASE"
    curl -fL --retry 8 --retry-delay 5 --retry-all-errors -C - \
      "$HF_BASE/$f" -o "$WORK/hf/$f"
  fi
}

for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
         vocab.json merges.txt; do
  stage_file "$f"
done

# safetensors are too large to commit -> always pull from HF.
echo "  fetching model.safetensors from $HF_BASE"
curl -fL --retry 8 --retry-delay 5 --retry-all-errors -C - \
  "$HF_BASE/model.safetensors" -o "$WORK/hf/model.safetensors"

echo "=== Step 2: Convert HF weights to FP32 nntrainer bin ==="
python3 "$CONVERTER" \
    --model_path "$WORK/hf" \
    --output_name "$WORK/nntr_qwen3_0.6b_fp32.bin" \
    --data_type float32

echo "=== Step 3: Prepare FP32 stage directory ==="
mkdir -p "$WORK/stage"
cp "$WORK/hf/config.json" "$WORK/hf/generation_config.json" \
   "$WORK/hf/tokenizer.json" "$WORK/hf/tokenizer_config.json" \
   "$WORK/hf/vocab.json" "$WORK/hf/merges.txt" "$WORK/stage/"
cat > "$WORK/stage/nntr_config.json" <<'EOFJSON'
{
    "model_type": "CausalLM",
    "model_tensor_type": "FP32-FP32",
    "model_file_name": "nntr_qwen3_0.6b_fp32.bin",
    "fc_layer_dtype": "FP32",
    "embedding_dtype": "FP32",
    "lmhead_dtype": "FP32",
    "lora_rank": 0,
    "lora_alpha": 0,
    "lora_target": [],
    "bad_word_ids": [],
    "fsu": false,
    "fsu_lookahead": 2,
    "num_to_generate": 32,
    "init_seq_len": 1024,
    "max_seq_len": 2048,
    "batch_size": 1,
    "tokenizer_file": "tokenizer.json",
    "sample_input": "<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n<|im_start|>assistant\n"
}
EOFJSON
mv "$WORK/nntr_qwen3_0.6b_fp32.bin" "$WORK/stage/"

echo "=== Step 4: Quantize to Q4_0 fc + Q6_K embed + Q6_K lmhead ==="
mkdir -p "$OUT_DIR"
"$QUANTIZER" "$WORK/stage" \
    --fc_dtype Q4_0 --embd_dtype Q6_K --lmhead_dtype Q6_K \
    --output_bin nntr_qwen3_0.6b_w4e6a32.bin \
    -o "$OUT_DIR"

echo "=== Step 5: Copy HF config + tokenizer assets into bundle ==="
for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do
    [ -f "$WORK/hf/$f" ] && cp "$WORK/hf/$f" "$OUT_DIR/"
done

echo "=== Step 6: Split bin into 95 MB parts ==="
(
  cd "$OUT_DIR" && \
  split -b 95000000 -a 2 nntr_qwen3_0.6b_w4e6a32.bin nntr_qwen3_0.6b_w4e6a32.bin.part_
)

echo "=== Step 7: Regenerate SHA256SUMS ==="
(
  cd "$OUT_DIR" && sha256sum \
    config.json generation_config.json merges.txt nntr_config.json \
    nntr_qwen3_0.6b_w4e6a32.bin.part_?? \
    tokenizer.json tokenizer_config.json vocab.json > SHA256SUMS
)

echo "=== Step 8: Sanity check against committed expected bin hash ==="
EXPECTED="eed53478ffd0d72241efde66bf505de74b3c63ff12f5bf0e363a0329f8c0ee2a"
ACTUAL=$(sha256sum "$OUT_DIR/nntr_qwen3_0.6b_w4e6a32.bin" | awk '{print $1}')
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "WARNING: rebuilt bin hash differs from combine.sh expectation." >&2
  echo "  expected: $EXPECTED" >&2
  echo "  got:      $ACTUAL"   >&2
  echo "  If you intentionally changed dtypes / quantizer version, update" >&2
  echo "  EXPECTED_SHA256 in combine.sh and re-run this recipe."        >&2
fi

echo "=== Done ==="
echo "Output directory: $OUT_DIR"
ls -lh "$OUT_DIR"
