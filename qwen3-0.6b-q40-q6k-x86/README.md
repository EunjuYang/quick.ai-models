# qwen3-0.6b-q40-q6k-x86

Pre-quantized Qwen3-0.6B for x86\_64, with:

- `fc_layer_dtype`: **Q4\_0**
- `embedding_dtype`: **Q6\_K**
- `lmhead_dtype`: **Q6\_K**

All three are intentional. Qwen3-0.6B has `tie_word_embeddings=true`, and the
nntrainer tied-embedding layer (`layers/tie_word_embedding.cpp` in Quick.AI,
`Applications/CausalLM/layers/tie_word_embedding.cpp` in nntrainer) currently
accepts only Q6\_K or FP32 weights on the tied path. Q4\_0 embedding / LM head
throws `Tieword embedding is not supported yet for the data type` at the first
decode step, which is why the sibling `qwen3-0.6b-q40-x86` bundle cannot run
inference end-to-end and this bundle exists.

## Contents (committed)

- `nntr_config.json` - runtime config (dtypes, seq lengths, tokenizer path)
- `combine.sh` - reassembles `nntr_qwen3_0.6b_w4e6a32.bin` from `.part_*`
- `SHA256SUMS` - checksums for every asset **including** the split bin parts

## Contents (not committed, rebuild locally)

The 4 bin parts (`nntr_qwen3_0.6b_w4e6a32.bin.part_aa` ... `part_ad`, totalling
~359 MB) and the HuggingFace config + tokenizer assets are not checked in.
Rebuild them with:

```bash
# Option A - nntrainer pipeline
export NNTRAINER=/path/to/nntrainer
../scripts/convert_qwen3_0.6b_q6k_lmhead.sh

# Option B - Quick.AI's quantize binary (if you already built Quick.AI)
export QUICK_DOT_AI_QUANTIZE=/path/to/Quick.AI/build/quick_dot_ai_quantize
../scripts/convert_qwen3_0.6b_q6k_lmhead.sh
```

Both invocations populate this directory with the 4 split parts plus the HF
assets so that `combine.sh` and `sha256sum -c SHA256SUMS` both succeed.

## Expected bin SHA256

```
eed53478ffd0d72241efde66bf505de74b3c63ff12f5bf0e363a0329f8c0ee2a  nntr_qwen3_0.6b_w4e6a32.bin
```
