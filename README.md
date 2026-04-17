# nntrainer-causallm-models

Pre-quantized CausalLM model weights for [nntrainer](https://github.com/EunjuYang/nntrainer) CI and benchmarks.

## Model Catalog

| Directory | Model | Platform | Quantization | Embedding | Source | Bin Size |
|-----------|-------|----------|-------------|-----------|--------|----------|
| `qwen3-0.6b-q40-x86` | Qwen3-0.6B | x86_64 Linux | Q4\_0 (FC) | Q4\_0 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | ~404 MB |

## Q4\_0 Platform Lock

Q4\_0 quantization produces **platform-specific** binary formats.
An x86-quantized `.bin` is **NOT compatible** with ARM, and vice versa.
The directory suffix (`-x86`, `-arm`) encodes the target architecture.

## Storage Format

Large `.bin` files are split into ~95 MB parts (`.bin.part_aa`, `.bin.part_ab`, ...)
to stay under GitHub's 100 MB per-file limit. Each model directory includes a
`combine.sh` script to reassemble and verify the full binary.

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

## Reproducing the Model

See `scripts/convert_qwen3_0.6b.sh` for the full pipeline:
HuggingFace download -> `weight_converter.py` (FP32) -> `nntr_quantize` (Q4\_0).

The script requires a locally-built nntrainer with `-Denable-transformer=true`.

## License

Model weights are subject to their upstream license (see respective HuggingFace model cards).
CI tooling in this repository is Apache-2.0.
