# nntrainer-causallm-models

Pre-quantized CausalLM model weights for [nntrainer](https://github.com/EunjuYang/nntrainer) CI and benchmarks.

## Model Catalog

| Directory | Model | Platform | FC | Embedding | LM head | Tied | Source | Bin Size |
|-----------|-------|----------|----|-----------|---------|------|--------|----------|
| `qwen3-0.6b-q40-x86` | Qwen3-0.6B | x86\_64 Linux | Q4\_0 | Q4\_0 | Q4\_0 | yes\* | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | ~404 MB |
| `qwen3-0.6b-q40-q6k-x86` | Qwen3-0.6B | x86\_64 Linux | Q4\_0 | Q6\_K | Q6\_K | yes | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | ~359 MB |

\* `qwen3-0.6b-q40-x86` is tied in the HuggingFace config but currently cannot
run inference with nntrainer's `tie_word_embedding` layer because the tied path
only accepts Q6\_K or FP32 weights. Use `qwen3-0.6b-q40-q6k-x86` when you need
tied inference to work end-to-end (Quick.AI unit tests, nntrainer CausalLM
smoke tests, etc.).

## Q4\_0 Platform Lock

Q4\_0 quantization produces **platform-specific** binary formats.
An x86-quantized `.bin` is **NOT compatible** with ARM, and vice versa.
The directory suffix (`-x86`, `-arm`) encodes the target architecture.

## Storage Format

Large `.bin` files are split into ~95 MB parts (`.bin.part_aa`, `.bin.part_ab`, ...)
to stay under GitHub's 100 MB per-file limit. Each model directory includes a
`combine.sh` script to reassemble and verify the full binary.

Bin **parts are not always pre-committed** for bandwidth reasons. When a
directory ships only metadata (`combine.sh`, `SHA256SUMS`, `nntr_config.json`,
tokenizer files) you can rebuild the parts locally by running the matching
script under `scripts/`.

## Usage in CI

```bash
git clone --depth 1 --branch main \
    https://github.com/eunjuyang/nntrainer-causallm-models.git models

# Reassemble the weight binary
cd models/qwen3-0.6b-q40-x86
chmod +x combine.sh && ./combine.sh

# Verify integrity (optional)
sha256sum -c SHA256SUMS
```

Then run inference:
```bash
./build/Applications/CausalLM/nntr_causallm models/qwen3-0.6b-q40-x86
```

## Reproducing the Models

| Directory | Recipe |
|-----------|--------|
| `qwen3-0.6b-q40-x86` | `scripts/convert_qwen3_0.6b.sh` |
| `qwen3-0.6b-q40-q6k-x86` | `scripts/convert_qwen3_0.6b_q6k_lmhead.sh` |

The Q4\_0 recipe requires a locally-built nntrainer with `-Denable-transformer=true`.
The Q6\_K-lmhead recipe can use either nntrainer's `nntr_quantize` or Quick.AI's
`quick_dot_ai_quantize`, both of which accept `--fc_dtype`, `--embd_dtype` and
`--lmhead_dtype`.

## License

Model weights are subject to their upstream license (see respective HuggingFace model cards).
CI tooling in this repository is Apache-2.0.
