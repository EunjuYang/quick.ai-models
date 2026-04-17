#!/bin/bash
# Reproducible recipe to generate Qwen3-0.6B Q4_0 x86 weights for nntrainer CI.
#
# Requirements:
#   - x86_64 Linux host (Q4_0 is platform-specific)
#   - Python 3 with: torch, transformers, huggingface_hub
#   - Built nntrainer with -Denable-transformer=true
#   - NNTRAINER env var pointing at the nntrainer checkout root
#
# Usage:
#   export NNTRAINER=/path/to/nntrainer
#   ./convert_qwen3_0.6b.sh

set -euo pipefail

: "${NNTRAINER:?Set NNTRAINER to the nntrainer repo root}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "=== Step 1: Download Qwen3-0.6B from HuggingFace ==="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-0.6B', local_dir='$WORK/hf')
"

echo "=== Step 2: Convert HF weights to nntrainer FP32 bin ==="
python3 "$NNTRAINER/Applications/CausalLM/res/qwen3/qwen3-0.6b/weight_converter.py" \
    --model_path "$WORK/hf" \
    --output_name "$WORK/nntr_qwen3_0.6b_fp32.bin" \
    --data_type float32

echo "=== Step 3: Prepare FP32 stage directory ==="
mkdir -p "$WORK/stage"
cp "$WORK/hf/config.json" "$WORK/hf/generation_config.json" "$WORK/stage/"
cat > "$WORK/stage/nntr_config.json" << 'EOFJSON'
{
    "model_type": "CausalLM",
    "model_tensor_type": "FP32-FP32",
    "model_file_name": "nntr_qwen3_0.6b_fp32.bin",
    "fc_layer_dtype": "FP32",
    "embedding_dtype": "FP32",
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

echo "=== Step 4: Quantize to Q4_0 (x86) ==="
OUT_DIR="$REPO_ROOT/qwen3-0.6b-q40-x86"
"$NNTRAINER/build/Applications/CausalLM/nntr_quantize" "$WORK/stage" \
    --fc_dtype Q4_0 --embd_dtype Q4_0 --lmhead_dtype Q4_0 \
    --output_bin nntr_qwen3_0.6b_w4e4a32.bin \
    -o "$OUT_DIR"

echo "=== Step 5: Copy HF config and tokenizer files ==="
for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do
    [ -f "$WORK/hf/$f" ] && cp "$WORK/hf/$f" "$OUT_DIR/"
done

echo "=== Step 6: Generate checksums ==="
(cd "$OUT_DIR" && sha256sum * > SHA256SUMS)

echo "=== Done ==="
echo "Output directory: $OUT_DIR"
ls -lh "$OUT_DIR"
