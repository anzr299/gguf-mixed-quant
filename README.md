# gguf-mixed-quant
> **_Better accuracy, same size._**

Sensitivity-driven mixed-precision GGUF quantization for [llama.cpp](https://github.com/ggerganov/llama.cpp), powered by [NNCF](https://github.com/openvinotoolkit/nncf).

Standard GGUF quantization assigns precision by hand tuning it. This tool uses **per-layer sensitivity analysis** to dyanmically steal bits from layers that don't need them and give them to layers that do like Robin Hood.

---

## Get Started (60 seconds)

**Prerequisites:** Python ≥ 3.10 and a working [llama.cpp](https://github.com/ggerganov/llama.cpp) build (you need `llama-quantize` and `convert_hf_to_gguf.py`).

```bash
# 1. Install
pip install git+https://github.com/anzr299/gguf-mixed-quant.git

# 2. Run (one command — handles everything)
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --metric max_activation_variance \
    --dataset wikitext \
    --preset Q4_K_M \
    --output my-model-q4km.gguf
```

That's it. The output GGUF has **better perplexity** than standard Q4_K_M because insensitive layers get downgraded to IQ4_XS and the saved bits promote sensitive layers to Q5_K/Q6_K.

> **Note:** Everything runs on CPU by default — no GPU required. Data-aware metrics like `max_activation_variance` give the best results but take a few minutes (~5-10 min) for calibration.

### Already have an F16 GGUF?

Skip the HF→GGUF conversion step (saves time on large models):

```bash
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --metric max_activation_variance \
    --dataset wikitext \
    --preset Q4_K_M \
    --f16-gguf /path/to/my-model-f16.gguf \
    --output my-model-q4km.gguf
```

### Want the fastest option? Use the data-free metric (no calibration data needed)

```bash
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --metric weight_quantization_error \
    --preset Q4_K_M \
    --output my-model-q4km.gguf
```

---

## Key Results

Llama 3.2 1B Instruct, wikitext-2 perplexity (lower is better):

| Method | BPW | Size | PPL |
|--------|-----|------|-----|
| llama.cpp Q4_K_M baseline | 5.18 | 763 MB | 12.495 |
| Unsloth UD-Q4_K_XL | 5.35 | 788 MB | 12.335 |
| **gguf-mixed-quant (Robin Hood)** | **5.21** | **796 MB** | **12.292** |

Robin Hood beats both the standard baseline and Unsloth Dynamic 2.0.

## How It Works

```
┌─────────────────┐     ┌──────────────────────┐     ┌───────────────────┐
│  HuggingFace    │────▶│  NNCF Sensitivity    │────▶│  Robin Hood       │
│  Model (FP16)   │     │  Analysis            │     │  Type Assignment  │
└─────────────────┘     └──────────────────────┘     └───────┬───────────┘
                                                             │
                        ┌──────────────────────┐             │
                        │  Quantized GGUF      │◀────────────┘
                        │  (llama-quantize)     │
                        └──────────────────────┘
```

1. **Compute sensitivity scores** — NNCF measures how much each layer's output degrades under quantization
2. **Rank layers** — most-sensitive layers need higher precision, least-sensitive can tolerate lower
3. **Robin Hood assignment** — downgrade insensitive layers to IQ4_XS, upgrade sensitive layers to Q5_K/Q6_K, within the same bit budget
4. **Quantize** — apply per-tensor type overrides via `llama-quantize --tensor-type-file`

The result uses 4 types (IQ4_XS, Q4_K_M, Q5_K_S, Q6_K) instead of the standard 2 (Q4_K_M, Q6_K), distributing bits where they matter most.

## Installation

```bash
# Option A: pip install directly from GitHub
pip install git+https://github.com/anzr299/gguf-mixed-quant.git

# Option B: editable install (for development)
git clone https://github.com/anzr299/gguf-mixed-quant.git
cd gguf-mixed-quant
pip install -e .
```

**Requirements**: Python ≥ 3.10, [llama.cpp](https://github.com/ggerganov/llama.cpp) built with `llama-quantize` (auto-detected from PATH, or pass `--llama-cpp /path/to/llama.cpp`).

## More Examples

### Different presets

```bash
# Q3_K_M — aggressive compression
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --metric max_activation_variance \
    --dataset wikitext \
    --preset Q3_K_M \
    --output model-q3km.gguf

# Q5_K_M — high quality
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --metric max_activation_variance \
    --dataset wikitext \
    --preset Q5_K_M \
    --output model-q5km.gguf
```

### Different calibration datasets

```bash
# Reasoning data for math-focused models
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --metric max_activation_variance \
    --dataset reasoning \
    --preset Q4_K_M \
    --output model-math.gguf
```

### Step-by-step (compute scores, then quantize separately)

```bash
# Step 1: Compute sensitivity and export overrides
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --metric max_activation_variance \
    --dataset wikitext \
    --preset Q4_K_M \
    --output-format llama-quantize-args \
    --output overrides.txt

# Step 2: Convert model to F16 GGUF (if you don't have one)
python llama.cpp/convert_hf_to_gguf.py meta-llama/Llama-3.2-1B-Instruct \
    --outfile model-f16.gguf --outtype f16

# Step 3: Quantize with overrides
llama-quantize --tensor-type-file overrides.txt model-f16.gguf model-mixed.gguf Q4_K_M
```

### Exploring sensitivity scores (analysis only)

```bash
# Output as table to inspect per-layer assignments
gguf-mixed-quant \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --metric max_activation_variance \
    --dataset wikitext \
    --preset Q4_K_M \
    --output-format table

# List all available metrics
gguf-mixed-quant --list-metrics

# List all presets
gguf-mixed-quant --list-presets

# List named datasets
gguf-mixed-quant --list-datasets
```

## Sensitivity Metrics

| Metric | Data | Description |
|--------|:---:|-------------|
| `weight_quantization_error` | No | Inverted 8-bit quantization noise per layer |
| `hessian_input_activation` | Yes | HAWQ: Hessian trace × quantization error |
| `mean_activation_variance` | Yes | Mean activation variance × quantization error |
| `max_activation_variance` | Yes | **Best results.** Max activation variance × quantization error |
| `mean_activation_magnitude` | Yes | Mean activation magnitude × quantization error |
| `yaqa_hessian_kronecker` | Yes | YAQA: Kronecker-factored Hessian sensitivity |

## Calibration Datasets

| Name | Description |
|------|-------------|
| `wikitext` | Wikipedia text (general language modeling) |
| `nemotron` | Nemotron SFT mix (math, science, chat, code) |
| `reasoning` | GSM8K math reasoning chains |
| `coding` | Python code generation outputs |
| `contextual` | CNN/DailyMail long news articles |

You can also pass any HuggingFace dataset name directly via `--dataset`.

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
                          Robin Hood mode is the default — steals bits from insensitive layers

Output:
  --output PATH           Output file path
  --output-format FMT     json | llama-quantize-args | table | gguf

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
