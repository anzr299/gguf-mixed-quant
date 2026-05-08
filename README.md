# gguf-mixed-quant

Sensitivity-driven mixed-precision GGUF quantization for [llama.cpp](https://github.com/ggerganov/llama.cpp), powered by [NNCF](https://github.com/openvinotoolkit/nncf).

Unlike standard GGUF quantization (which assigns precision by layer *position* — e.g. first/last layers get bumped), this tool uses actual **per-layer sensitivity analysis** to decide which layers deserve higher precision. 
The idea is better accuracy at the same size.

## Key Results

Llama 3.2 1B, wikitext-2 perplexity (lower is better), **same file size**:

| Method | Preset | Size | PPL |
|--------|--------|------|-----|
| llama.cpp baseline | Q4_K_M | 771 MB | 9.2251 |
| **gguf-mixed-quant (refine)** | Q4_K_M | 771 MB | **9.1819** |
| llama.cpp baseline | Q3_K_M | 659 MB | 10.5502 |
| **gguf-mixed-quant (refine)** | Q3_K_M | 659 MB | **10.4896** |

## How It Works

```
┌─────────────────┐     ┌──────────────────────┐     ┌───────────────────┐
│  HuggingFace    │────▶│  NNCF Sensitivity    │────▶│  Assign GGUF      │
│  Model (FP16)   │     │  Analysis            │     │  Types by Score   │
└─────────────────┘     └──────────────────────┘     └───────┬───────────┘
                                                             │
                        ┌──────────────────────┐             │
                        │  Quantized GGUF      │◀────────────┘
                        │  (llama-quantize)     │
                        └──────────────────────┘
```

1. **Compute sensitivity scores** — NNCF measures how much each layer's output degrades under quantization
2. **Rank layers** — most-sensitive layers need higher precision, least-sensitive can tolerate lower
3. **Assign quant types** — map sensitivity rankings to GGUF types (Q2_K through Q8_0)
4. **Quantize** — apply per-tensor type overrides via `llama-quantize --tensor-type-file`

### Refine Mode (recommended)

Instead of assigning types from scratch, `--refine` takes llama.cpp's own baseline assignment (e.g. Q4_K_M) and **redistributes** the bump slots by sensitivity. Same total bit budget, better allocation.

### Dip Mode

`--dip-fraction 0.1` goes further: it **downgrades** the 10% least-sensitive layers one tier below the baseline (e.g. Q4_K → Q3_K) and uses the saved bits to **promote** more sensitive layers upward. This trades off insensitive layers for quality-critical ones.

## Installation

```bash
git clone https://github.com/anzr299/gguf-mixed-quant.git
cd gguf-mixed-quant
pip install -e .
```

**Requirements**: Python ≥ 3.10, [llama.cpp](https://github.com/ggerganov/llama.cpp) built with `llama-quantize`.

## Quick Start

### One-command quantization (recommended)

```bash
# Refine Q4_K_M — same size as baseline, better quality
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B \
    --metric max_activation_variance \
    --dataset wikitext \
    --preset Q4_K_M \
    --refine \
    --output /tmp/llama-1b-refined-q4km.gguf

# With dip: downgrade 10% least-sensitive, promote sensitive layers
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B \
    --metric max_activation_variance \
    --dataset wikitext \
    --preset Q4_K_M \
    --refine \
    --dip-fraction 0.1 \
    --output /tmp/llama-1b-refined-dip-q4km.gguf
```

### Step-by-step (compute scores, then quantize separately)

```bash
# Step 1: Compute sensitivity and export overrides
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B \
    --metric weight_quantization_error \
    --preset Q4_K_M \
    --output-format llama-quantize-args \
    --output overrides.txt

# Step 2: Convert model to F16 GGUF
python llama.cpp/convert_hf_to_gguf.py meta-llama/Llama-3.2-1B \
    --outfile model-f16.gguf --outtype f16

# Step 3: Quantize with overrides
llama-quantize --tensor-type-file overrides.txt model-f16.gguf model-mixed.gguf Q4_K_M
```

### Use an existing F16 GGUF

```bash
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B \
    --metric max_activation_variance \
    --dataset wikitext \
    --preset Q3_K_M \
    --refine \
    --f16-gguf /tmp/llama-3.2-1b-f16.gguf \
    --output /tmp/llama-1b-refined-q3km.gguf
```

## Tutorials

### Tutorial 1: Data-free quantization (fastest)

No calibration data needed — uses weight quantization error as a proxy for sensitivity.

```bash
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B \
    --metric weight_quantization_error \
    --preset Q4_K_M \
    --refine \
    --output model-q4km-refined.gguf
```

### Tutorial 2: Data-aware quantization (best quality)

Uses `max_activation_variance` with calibration data. This is the metric that produced the best perplexity results in our benchmarks.

```bash
# Use wikitext for general-purpose models
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B \
    --metric max_activation_variance \
    --dataset wikitext \
    --subset-size 128 \
    --preset Q4_K_M \
    --refine \
    --output model-q4km-mav.gguf

# Use reasoning data for math-focused models
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B \
    --metric max_activation_variance \
    --dataset reasoning \
    --preset Q4_K_M \
    --refine \
    --output model-q4km-math.gguf
```

### Tutorial 3: Aggressive compression with Q3_K_M

```bash
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B \
    --metric max_activation_variance \
    --dataset wikitext \
    --preset Q3_K_M \
    --refine \
    --dip-fraction 0.1 \
    --output model-q3km-refined.gguf
```

### Tutorial 4: Exploring sensitivity scores (analysis only)

```bash
# Output as table to inspect per-layer assignments
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B \
    --metric max_activation_variance \
    --dataset wikitext \
    --preset Q4_K_M \
    --output-format table

# Export as JSON for scripting
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B \
    --metric weight_quantization_error \
    --preset Q4_K_M \
    --output-format json \
    --output plan.json

# List all available metrics
gguf-mixed-quant --list-metrics

# List all presets
gguf-mixed-quant --list-presets

# List named datasets
gguf-mixed-quant --list-datasets
```

## Python API

```python
from gguf_mixed_quant import compute_sensitivity, assign_gguf_types, export_overrides
from gguf_mixed_quant.precision_assignment import (
    assign_gguf_types_preset,
    refine_baseline,
)
from gguf_mixed_quant.baseline import get_baseline_assignments, baseline_to_map
from pathlib import Path

# --- Example 1: Simple two-level assignment ---
scores = compute_sensitivity(
    model_id="meta-llama/Llama-3.2-1B",
    metric="weight_quantization_error",
)

plan = assign_gguf_types(scores, ratio=0.8, primary_type="Q4_K_M", backup_type="Q6_K")
print(plan.summary())

overrides = export_overrides(plan, format="llama-quantize-args")
print(overrides)

# --- Example 2: Preset-based (matches llama.cpp tier structure) ---
scores = compute_sensitivity(
    model_id="meta-llama/Llama-3.2-1B",
    metric="max_activation_variance",
    dataset_name="wikitext",
    subset_size=128,
)

plan = assign_gguf_types_preset(scores, preset_name="Q4_K_M")
print(plan.summary())
print(plan.avg_bpw)
print(plan.type_distribution)

# --- Example 3: Refine llama.cpp baseline ---
llama_cpp = Path("/tmp/llama_cpp_output/llama.cpp")
f16_gguf = Path("/tmp/llama-3.2-1b-f16.gguf")

baseline = get_baseline_assignments(f16_gguf, "Q4_K_M", llama_cpp)
baseline_map = baseline_to_map(baseline)

plan = refine_baseline(
    baseline_map=baseline_map,
    sensitivity_result=scores,
    dip_fraction=0.1,  # downgrade 10% least-sensitive, promote sensitive
)
print(plan.summary())

# Export and quantize
export_overrides(plan, format="llama-quantize-args", output_path="overrides.txt")
```

## Sensitivity Metrics

| Metric | Data | Description |
|--------|:---:|-------------|
| `weight_quantization_error` | No | Inverted 8-bit quantization noise per layer |
| `hessian_input_activation` | Yes | HAWQ: Hessian trace × quantization error |
| `mean_activation_variance` | Yes | Mean activation variance × quantization error |
| `max_activation_variance` | Yes | **Best results.** Max activation variance × quantization error |
| `mean_activation_magnitude` | Yes | Mean activation magnitude × quantization error |

## Calibration Datasets

| Name | Description |
|------|-------------|
| `wikitext` | Wikipedia text (general language modeling) |
| `nemotron` | Nemotron SFT mix (math, science, chat, code) |
| `reasoning` | GSM8K math reasoning chains |
| `coding` | Python code generation outputs |
| `contextual` | CNN/DailyMail long news articles |

You can also pass any HuggingFace dataset name directly via `--dataset`.

## GGUF Quantization Types

| Type | BPW | Description |
|------|-----|-------------|
| Q2_K | 2.63 | 2-bit K-quants |
| Q3_K_S | 3.44 | 3-bit K-quants (small) |
| Q3_K_M | 3.91 | 3-bit K-quants (medium) |
| Q4_K_S | 4.59 | 4-bit K-quants (small) |
| Q4_K_M | 4.85 | 4-bit K-quants (medium) |
| Q5_K_S | 5.54 | 5-bit K-quants (small) |
| Q5_K_M | 5.69 | 5-bit K-quants (medium) |
| Q6_K | 6.56 | 6-bit K-quants |
| Q8_0 | 8.50 | 8-bit quantization |

## CLI Reference

```
gguf-mixed-quant [OPTIONS]

Required:
  --model MODEL           HuggingFace model ID or local path

Sensitivity:
  --metric METRIC         Sensitivity metric (default: weight_quantization_error)
  --dataset DATASET       Calibration dataset for data-aware metrics
  --subset-size N         Number of calibration samples (default: 128)

Quantization:
  --preset PRESET         Multi-level preset: Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, etc.
  --ratio RATIO           Fraction of weights for lower precision (default: 0.8)
  --primary-type TYPE     Low-precision type (default: Q4_K_M)
  --backup-type TYPE      High-precision type (default: Q6_K)
  --refine                Refine llama.cpp baseline instead of assigning from scratch
  --dip-fraction FRAC     Downgrade fraction of insensitive layers (0.0-1.0)

Output:
  --output PATH           Output file path
  --output-format FMT     json | llama-quantize-args | table | gguf
  --quantize              Run full pipeline (compute → convert → quantize)

Paths:
  --llama-cpp PATH        Path to llama.cpp directory
  --f16-gguf PATH         Path to existing F16 GGUF (skip conversion)

Info:
  --list-metrics          Show available sensitivity metrics
  --list-presets          Show available quantization presets
  --list-datasets         Show available calibration datasets
```

## License

Apache-2.0
