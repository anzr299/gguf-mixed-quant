# gguf-mixed-quant

Mixed-precision GGUF quantization for llama.cpp using [NNCF](https://github.com/openvinotoolkit/nncf) sensitivity metrics.

## Overview

This tool uses NNCF's mixed-precision algorithms to compute per-layer sensitivity scores for LLMs, then maps those scores to different GGUF quantization types. Less sensitive layers get more aggressive quantization (e.g., Q2_K, Q3_K), while more sensitive layers are kept at higher precision (e.g., Q6_K, Q8_0).

## Features

- **Multiple sensitivity metrics**: Data-free (weight quantization error) and data-aware (HAWQ, mean/max activation variance, YAQA Hessian Kronecker)
- **Flexible precision mapping**: Map sensitivity scores to any GGUF quantization type
- **Two output modes**:
  - Generate `--override-tensor-type` args for llama.cpp's `llama-quantize` CLI
  - Direct Python-based GGUF quantization using the `gguf` library
- **Configurable ratio**: Control what percentage of layers get lower precision

## Installation

```bash
pip install -e .
```

## Quick Start

### Generate llama-quantize overrides

```bash
# Data-free (no calibration data needed)
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B \
    --metric weight_quantization_error \
    --ratio 0.8 \
    --primary-type Q4_K_M \
    --backup-type Q6_K \
    --output overrides.json

# Data-aware (uses calibration dataset for better accuracy)
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B \
    --metric hessian_input_activation \
    --dataset wikitext \
    --ratio 0.8 \
    --num-levels 4 \
    --output overrides.json
```

### Use with llama-quantize

```bash
# First convert to F16 GGUF
python convert_hf_to_gguf.py meta-llama/Llama-3.2-1B --outfile model-f16.gguf

# Then quantize with overrides
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B \
    --metric weight_quantization_error \
    --ratio 0.8 \
    --output-format llama-quantize-args \
    --output overrides.txt

# Apply (paste the generated args)
llama-quantize model-f16.gguf model-mixed.gguf Q4_K_M $(cat overrides.txt)
```

### Direct GGUF quantization (Python)

```bash
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B \
    --metric hessian_input_activation \
    --dataset wikitext \
    --ratio 0.8 \
    --num-levels 4 \
    --output-format gguf \
    --output model-mixed.gguf
```

### Python API

```python
from gguf_mixed_quant import compute_sensitivity, assign_gguf_types, export_overrides

# Compute sensitivity scores
scores = compute_sensitivity(
    model_id="meta-llama/Llama-3.2-1B",
    metric="weight_quantization_error",
)

# Assign GGUF types based on scores
assignments = assign_gguf_types(
    scores,
    ratio=0.8,
    primary_type="Q4_K_M",
    backup_type="Q6_K",
)

# Export as llama-quantize overrides
overrides = export_overrides(assignments, format="llama-quantize-args")
print(overrides)
```

## Sensitivity Metrics

| Metric | Data Required | Description |
|--------|:---:|-------------|
| `weight_quantization_error` | No | Inverted 8-bit quantization noise per layer |
| `hessian_input_activation` | Yes | HAWQ: Hessian trace × quantization error |
| `mean_activation_variance` | Yes | Mean activation variance × quantization error |
| `max_activation_variance` | Yes | Max activation variance × quantization error |
| `mean_activation_magnitude` | Yes | Mean activation magnitude × quantization error |

## GGUF Quantization Types (ordered by bits-per-weight)

| Type | BPW | Description |
|------|-----|-------------|
| Q2_K | ~2.6 | 2-bit with K-quants |
| Q3_K_S | ~3.4 | 3-bit K-quants (small) |
| Q3_K_M | ~3.9 | 3-bit K-quants (medium) |
| Q4_K_S | ~4.6 | 4-bit K-quants (small) |
| Q4_K_M | ~4.8 | 4-bit K-quants (medium) |
| Q5_K_S | ~5.5 | 5-bit K-quants (small) |
| Q5_K_M | ~5.7 | 5-bit K-quants (medium) |
| Q6_K | ~6.6 | 6-bit K-quants |
| Q8_0 | ~8.5 | 8-bit |

## How It Works

1. **Load model** from HuggingFace in PyTorch
2. **Compute sensitivity scores** using NNCF's mixed-precision algorithms
3. **Rank layers** by sensitivity (ascending)
4. **Assign quantization types** — least sensitive layers get more aggressive quantization
5. **Export** as llama-quantize CLI args, JSON config, or direct GGUF file

## License

Apache-2.0
