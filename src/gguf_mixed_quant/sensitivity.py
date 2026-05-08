"""Compute per-layer sensitivity scores using NNCF's mixed-precision algorithms."""

from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import nncf
from nncf.parameters import CompressWeightsMode, SensitivityMetric


# Map user-friendly metric names to NNCF enums
METRIC_MAP: dict[str, SensitivityMetric] = {
    "weight_quantization_error": SensitivityMetric.WEIGHT_QUANTIZATION_ERROR,
    "hessian_input_activation": SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
    "mean_activation_variance": SensitivityMetric.MEAN_ACTIVATION_VARIANCE,
    "max_activation_variance": SensitivityMetric.MAX_ACTIVATION_VARIANCE,
    "mean_activation_magnitude": SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE,
}

DATA_FREE_METRICS = {"weight_quantization_error"}


@dataclass
class LayerSensitivity:
    """Sensitivity score for a single layer."""

    layer_name: str
    score: float
    num_weights: int


@dataclass
class SensitivityResult:
    """Collection of per-layer sensitivity scores."""

    model_id: str
    metric: str
    layers: list[LayerSensitivity]

    @property
    def scores(self) -> dict[str, float]:
        """Return a dict mapping layer names to scores."""
        return {layer.layer_name: layer.score for layer in self.layers}

    @property
    def sorted_layers(self) -> list[LayerSensitivity]:
        """Return layers sorted by score (ascending = least sensitive first)."""
        return sorted(self.layers, key=lambda x: x.score)


def _transform_fn(data: dict, tokenizer, text_key: str = "text") -> dict:
    """Tokenize text data for calibration."""
    tokenized = tokenizer(data[text_key], return_tensors="pt")
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }


# Named dataset aliases for convenience
DATASET_ALIASES: dict[str, dict] = {
    "wikitext": {
        "path": "Salesforce/wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "test",
        "text_key": "text",
        "description": "Wikipedia text (general language modeling)",
    },
    "nemotron": {
        "path": "nvidia/Nemotron-Cascade-2-SFT-Data",
        "name": "__mixed_configs__",
        "split": "train",
        "text_key": "__messages__",
        "description": "Nemotron SFT mix (math, science, chat, code, instruction following)",
    },
    "reasoning": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "train",
        "text_key": "__concat_qa__",
        "description": "GSM8K math reasoning chains (question + solution)",
    },
    "coding": {
        "path": "iamtarun/python_code_instructions_18k_alpaca",
        "name": None,
        "split": "train",
        "text_key": "output",
        "description": "Python code generation outputs (18k Alpaca)",
    },
    "contextual": {
        "path": "ccdv/cnn_dailymail",
        "name": "3.0.0",
        "split": "train",
        "text_key": "article",
        "description": "CNN/DailyMail long news articles",
    },
}

# Nemotron configs to sample from (balanced mix of domains)
_NEMOTRON_CONFIGS = ["math", "science", "chat", "instruction_following", "swe"]


def list_available_datasets() -> dict[str, str]:
    """List available named dataset aliases."""
    return {name: info["description"] for name, info in DATASET_ALIASES.items()}


def _load_nemotron_mixed(subset_size: int):
    """Load a balanced mix of Nemotron configs, concatenating messages into text."""
    from datasets import Dataset, load_dataset

    per_config = max(1, subset_size // len(_NEMOTRON_CONFIGS))
    items = []

    for config in _NEMOTRON_CONFIGS:
        ds = load_dataset(
            "nvidia/Nemotron-Cascade-2-SFT-Data", config,
            split="train", streaming=True,
        )
        count = 0
        for item in ds:
            msgs = item.get("messages", [])
            text = "\n".join(m.get("content", "") or "" for m in msgs)
            if len(text.strip()) > 50:
                items.append({"__text__": text})
                count += 1
            if count >= per_config:
                break

    # Trim to exact subset_size
    items = items[:subset_size]
    return Dataset.from_list(items)


def _build_calibration_dataset(
    dataset_name: str,
    tokenizer,
    subset_size: int = 128,
) -> nncf.Dataset:
    """Build a calibration dataset from HuggingFace datasets."""
    from datasets import load_dataset

    # Check if it's a named alias
    if dataset_name in DATASET_ALIASES:
        alias = DATASET_ALIASES[dataset_name]
        text_key = alias["text_key"]

        # Special handling: Nemotron mixed configs (math, science, chat, code, etc.)
        if text_key == "__messages__":
            ds = _load_nemotron_mixed(subset_size)
            text_key = "__text__"
        else:
            load_name = alias.get("name")
            load_kwargs = {"split": alias["split"], "trust_remote_code": True}
            if load_name:
                ds = load_dataset(alias["path"], load_name, **load_kwargs)
            else:
                ds = load_dataset(alias["path"], **load_kwargs)

            # Special handling: concatenate multiple fields for richer signal
            if text_key == "__concat_qa__":
                # GSM8K: combine question + answer for full reasoning chain
                ds = ds.map(lambda x: {"__text__": x["question"] + "\n" + x["answer"]})
                text_key = "__text__"

            ds = ds.filter(lambda x: len(str(x.get(text_key, "")).strip()) > 10)
            ds = ds.select(range(min(subset_size, len(ds))))
    elif dataset_name in ("wikitext", "wikitext-2"):
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text_key = "text"
        ds = ds.filter(lambda x: len(x["text"].strip()) > 10)
        ds = ds.select(range(min(subset_size, len(ds))))
    else:
        # Generic HuggingFace dataset - assume "text" field
        ds = load_dataset(dataset_name, split="train")
        text_key = "text"
        ds = ds.filter(lambda x: len(x["text"].strip()) > 10)
        ds = ds.select(range(min(subset_size, len(ds))))

    return nncf.Dataset(ds, partial(_transform_fn, tokenizer=tokenizer, text_key=text_key))


def compute_sensitivity(
    model_id: str,
    metric: str = "weight_quantization_error",
    dataset_name: Optional[str] = None,
    subset_size: int = 128,
    group_size: int = 128,
    torch_dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> SensitivityResult:
    """
    Compute per-layer sensitivity scores for a HuggingFace model using NNCF.

    :param model_id: HuggingFace model ID or local path.
    :param metric: Sensitivity metric name (see METRIC_MAP).
    :param dataset_name: HuggingFace dataset for data-aware metrics.
    :param subset_size: Number of calibration samples.
    :param group_size: Quantization group size.
    :param torch_dtype: Model dtype for loading.
    :param device: Device to load model on.
    :return: SensitivityResult with per-layer scores.
    """
    if metric not in METRIC_MAP:
        valid = ", ".join(METRIC_MAP.keys())
        raise ValueError(f"Unknown metric: '{metric}'. Valid metrics: {valid}")

    sensitivity_metric = METRIC_MAP[metric]

    # Data-aware metrics require a dataset
    if metric not in DATA_FREE_METRICS and dataset_name is None:
        raise ValueError(f"Metric '{metric}' requires a calibration dataset. Pass --dataset.")

    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Build calibration dataset if needed
    calibration_dataset = None
    if dataset_name is not None:
        print(f"Building calibration dataset from: {dataset_name} ({subset_size} samples)")
        calibration_dataset = _build_calibration_dataset(dataset_name, tokenizer, subset_size)

    print(f"Computing sensitivity scores with metric: {metric}")

    from nncf.torch import wrap_model
    from nncf.quantization.algorithms.weight_compression.algorithm import WeightCompression
    from nncf.quantization.algorithms.weight_compression.mixed_precision import MIXED_PRECISION_CRITERIA
    from nncf.scopes import IgnoredScope
    from nncf.parameters import BackupMode

    # Wrap model for NNCF graph access
    example_input = _get_example_input(model, tokenizer, device)
    wrapped_model = wrap_model(model, example_input=example_input, trace_parameters=True)
    graph = wrapped_model.get_graph()

    # Create WC algo to extract weight params via its internal method
    wc_algo = WeightCompression(
        mode=CompressWeightsMode.INT4_SYM,
        ratio=0.8,
        group_size=group_size,
        ignored_scope=IgnoredScope(),
        all_layers=False,
        sensitivity_metric=sensitivity_metric,
        awq=False,
        subset_size=subset_size,
        scale_estimation=False,
        gptq=False,
        lora_correction=False,
        backup_mode=BackupMode.INT8_ASYM,
    )
    wc_algo.set_backend_entity(wrapped_model)

    # Get weight params using the public method
    all_weight_params, ratio_defining_params, _ = wc_algo.get_weight_compression_parameters(wrapped_model, graph)

    # Use ratio_defining_params (MatMul layers only, excludes embeddings/last layer)
    weight_params = ratio_defining_params

    # Compute sensitivity using the mixed precision criterion
    criterion_cls = MIXED_PRECISION_CRITERIA.get(sensitivity_metric)
    criterion = criterion_cls(ratio=0.8, subset_size=subset_size)
    criterion._set_backend_entity(wrapped_model)

    # For data-aware, we need statistics
    statistic_points = None
    if metric not in DATA_FREE_METRICS:
        # Build the activation-node-to-matmul map (keys are (node, port_id, channel_axis) tuples)
        matmul_nodes = [wp.node_with_weight for wp in weight_params]
        matmul_input_map = wc_algo.get_matmul_input_to_output_nodes_map(matmul_nodes, graph)
        statistic_points = criterion.get_statistic_points(wrapped_model, graph, matmul_input_map.keys())

        # Collect statistics using NNCF's factory
        from nncf.common.factory import StatisticsAggregatorFactory

        aggregator = StatisticsAggregatorFactory.create(wrapped_model, calibration_dataset)
        aggregator.stat_subset_size = subset_size
        for sp_key, sp_list in statistic_points.items():
            for sp in sp_list:
                aggregator.statistic_points.add_statistic_point(sp)
        aggregator.collect_statistics(wrapped_model, graph)

    # Calculate scores
    scores = criterion._calc_sensitivity(wrapped_model, graph, weight_params, statistic_points)

    # Build result
    layers = []
    for wp, score in zip(weight_params, scores):
        layers.append(LayerSensitivity(
            layer_name=wp.weight_name,
            score=float(score),
            num_weights=int(wp.num_weights),
        ))

    print(f"Computed scores for {len(layers)} layers")
    return SensitivityResult(model_id=model_id, metric=metric, layers=layers)


def _get_example_input(model, tokenizer, device: str) -> dict:
    """Generate an example input for model tracing."""
    text = "Hello, this is a test."
    inputs = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def list_available_metrics() -> dict[str, dict]:
    """List all available sensitivity metrics with descriptions."""
    return {
        "weight_quantization_error": {
            "requires_data": False,
            "description": "Inverted 8-bit quantization noise per layer (data-free baseline)",
        },
        "hessian_input_activation": {
            "requires_data": True,
            "description": "HAWQ: Average Hessian trace × Frobenius norm of quantization noise",
        },
        "mean_activation_variance": {
            "requires_data": True,
            "description": "Mean variance of input activations × quantization error",
        },
        "max_activation_variance": {
            "requires_data": True,
            "description": "Maximum variance of input activations × quantization error",
        },
        "mean_activation_magnitude": {
            "requires_data": True,
            "description": "Mean magnitude of input activations × quantization error",
        },
    }
