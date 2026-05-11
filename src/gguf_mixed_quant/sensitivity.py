"""Compute per-layer sensitivity scores using NNCF's mixed-precision algorithms.

Provides calibration dataset construction from named aliases (wikitext,
nemotron, reasoning, coding) with per-dataset defaults for sequence length
and subset size.  Data-aware metrics also auto-compute variance ratios
(max/mean activation variance per layer) used downstream for IQ/K-quant
sub-type assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

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

DATA_FREE_METRICS = {
    "weight_quantization_error",
}

_EPS = 1e-9


@dataclass
class LayerSensitivity:
    """Sensitivity score for a single layer."""

    layer_name: str
    score: float
    num_weights: int
    variance_ratio: float | None = None


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


def _transform_fn(
    data: dict,
    tokenizer,
    text_key: str = "text",
    seq_len: int | None = None,
    device: str = "cpu",
) -> dict:
    """Tokenize text data for calibration, optionally truncating to seq_len."""
    kwargs: dict = {"return_tensors": "pt"}
    if seq_len is not None:
        kwargs["max_length"] = seq_len
        kwargs["truncation"] = True
    tokenized = tokenizer(data[text_key], **kwargs)
    return {
        "input_ids": tokenized["input_ids"].to(device),
        "attention_mask": tokenized["attention_mask"].to(device),
    }


# Named dataset aliases for convenience
DATASET_ALIASES: dict[str, dict] = {
    "wikitext": {
        "path": "Salesforce/wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "test",
        "text_key": "text",
        "description": "Wikipedia text (general language modeling)",
        "default_seq_len": 256,
        "default_subset_size": 128,
    },
    "nemotron": {
        "path": "nvidia/Nemotron-Cascade-2-SFT-Data",
        "name": "__mixed_configs__",
        "split": "train",
        "text_key": "__messages__",
        "description": "Nemotron SFT mix (math, science, chat, swe, instruction following)",
        "default_seq_len": 8192,
        "default_subset_size": 32,
    },
    "reasoning": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "train",
        "text_key": "__concat_qa__",
        "description": "GSM8K math reasoning chains (question + solution)",
        "default_seq_len": 512,
        "default_subset_size": 128,
    },
    "coding": {
        "path": "iamtarun/python_code_instructions_18k_alpaca",
        "name": None,
        "split": "train",
        "text_key": "output",
        "description": "Python code generation outputs (18k Alpaca)",
        "default_seq_len": 1024,
        "default_subset_size": 64,
    },
}

# Nemotron configs to sample from (balanced mix of domains)
_NEMOTRON_CONFIGS = ["math", "science", "chat", "instruction_following", "swe"]


def list_available_datasets() -> dict[str, str]:
    """List available named dataset aliases with default parameters."""
    result = {}
    for name, info in DATASET_ALIASES.items():
        description = info["description"]
        default_seq = info.get("default_seq_len", "auto")
        default_subset = info.get("default_subset_size", 128)
        result[name] = f"{description} (seq_len={default_seq}, subset_size={default_subset})"
    return result


def _load_nemotron_mixed(subset_size: int) -> Dataset:
    """Load a balanced mix of Nemotron configs, concatenating messages into text."""
    from datasets import Dataset, load_dataset

    per_config = max(1, subset_size // len(_NEMOTRON_CONFIGS))
    items = []

    for config in _NEMOTRON_CONFIGS:
        config_dataset = load_dataset(
            "nvidia/Nemotron-Cascade-2-SFT-Data", config,
            split="train", streaming=True,
        )
        count = 0
        for item in config_dataset:
            msgs = item.get("messages", [])
            text = "\n".join(message.get("content", "") or "" for message in msgs)
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
    seq_len: int | None = None,
    device: str = "cpu",
) -> nncf.Dataset:
    """
    Build an nncf.Dataset from a named alias or generic HuggingFace dataset.

    Named aliases (see DATASET_ALIASES) handle special loading logic
    (Nemotron multi-config mixing, GSM8K Q+A concatenation).  Generic
    datasets are loaded with ``split='train'`` and a ``'text'`` column.

    :param dataset_name: Key in DATASET_ALIASES or any HF dataset identifier.
    :param tokenizer: HuggingFace tokenizer for the target model.
    :param subset_size: Number of samples to select.
    :param seq_len: Optional max token length passed to the tokenizer.
    :param device: Device to place tensors on.
    :return: An nncf.Dataset wrapping the tokenized samples.
    """
    from datasets import load_dataset

    # Check if it's a named alias
    if dataset_name in DATASET_ALIASES:
        alias = DATASET_ALIASES[dataset_name]
        text_key = alias["text_key"]

        # Special handling: Nemotron mixed configs (math, science, chat, swe, etc.)
        if text_key == "__messages__":
            dataset = _load_nemotron_mixed(subset_size)
            text_key = "__text__"
        else:
            load_name = alias.get("name")
            load_kwargs = {"split": alias["split"], "trust_remote_code": True}
            if load_name:
                dataset = load_dataset(alias["path"], load_name, **load_kwargs)
            else:
                dataset = load_dataset(alias["path"], **load_kwargs)

            # Special handling: concatenate multiple fields for richer signal
            if text_key == "__concat_qa__":
                # GSM8K: combine question + answer for full reasoning chain
                dataset = dataset.map(lambda x: {"__text__": x["question"] + "\n" + x["answer"]})
                text_key = "__text__"

            dataset = dataset.filter(lambda x: len(str(x.get(text_key, "")).strip()) > 10)
            dataset = dataset.select(range(min(subset_size, len(dataset))))
    else:
        # Generic HuggingFace dataset - assume "text" field
        dataset = load_dataset(dataset_name, split="train")
        text_key = "text"
        dataset = dataset.filter(lambda x: len(x["text"].strip()) > 10)
        dataset = dataset.select(range(min(subset_size, len(dataset))))

    return nncf.Dataset(
        dataset,
        partial(_transform_fn, tokenizer=tokenizer, text_key=text_key, seq_len=seq_len, device=device),
)


def compute_sensitivity(
    model_id: str,
    metric: str = "max_activation_variance",
    dataset_name: str | None = "wikitext",
    subset_size: int | None = None,
    seq_len: int | None = None,
    group_size: int = 128,
    torch_dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> SensitivityResult:
    """
    Compute per-layer sensitivity scores for a HuggingFace model using NNCF.

    :param model_id: HuggingFace model ID or local path.
    :param metric: Sensitivity metric name (see METRIC_MAP).
    :param dataset_name: HuggingFace dataset for data-aware metrics.
    :param subset_size: Number of calibration samples. None = use dataset default.
    :param seq_len: Max sequence length for tokenization. None = use dataset default.
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

    # Resolve per-dataset defaults for subset_size and seq_len
    if dataset_name is not None and dataset_name in DATASET_ALIASES:
        alias = DATASET_ALIASES[dataset_name]
        if subset_size is None:
            subset_size = alias.get("default_subset_size", 128)
        if seq_len is None:
            seq_len = alias.get("default_seq_len")
    if subset_size is None:
        subset_size = 128  # global fallback

    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Build calibration dataset if needed
    calibration_dataset = None
    if dataset_name is not None:
        seq_info = f", seq_len={seq_len}" if seq_len else ""
        print(f"Building calibration dataset from: {dataset_name} ({subset_size} samples{seq_info})")
        calibration_dataset = _build_calibration_dataset(
            dataset_name, tokenizer, subset_size, seq_len=seq_len, device=device,
        )

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

    # Create WC algo to extract weight compression parameters
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

    _, weight_params, _ = wc_algo.get_weight_compression_parameters(wrapped_model, graph)

    # Compute sensitivity using the mixed precision criterion
    criterion_cls = MIXED_PRECISION_CRITERIA.get(sensitivity_metric)
    criterion = criterion_cls(ratio=0.8, subset_size=subset_size)
    criterion._set_backend_entity(wrapped_model)

    statistic_points = None
    mean_var_criterion = None
    max_var_criterion = None
    mean_var_stat_points = None
    max_var_stat_points = None

    if metric not in DATA_FREE_METRICS:
        from nncf.quantization.algorithms.weight_compression.mixed_precision import (
            MeanVarianceCriterion,
            MaxVarianceCriterion,
        )
        from nncf.common.factory import StatisticsAggregatorFactory

        matmul_nodes = [wp.node_with_weight for wp in weight_params]
        matmul_input_map = wc_algo.get_matmul_input_to_output_nodes_map(matmul_nodes, graph)
        activation_keys = matmul_input_map.keys()

        statistic_points = criterion.get_statistic_points(wrapped_model, graph, activation_keys)

        # Set up both variance criteria for ratio computation,
        # reusing the primary criterion when it matches.
        def _make_criterion(cls):
            criterion_instance = cls(ratio=0.8, subset_size=subset_size)
            criterion_instance._set_backend_entity(wrapped_model)
            criterion_stat_points = criterion_instance.get_statistic_points(
                wrapped_model, graph, activation_keys,
            )
            return criterion_instance, criterion_stat_points

        if metric == "mean_activation_variance":
            mean_var_criterion, mean_var_stat_points = criterion, statistic_points
            max_var_criterion, max_var_stat_points = _make_criterion(MaxVarianceCriterion)
        elif metric == "max_activation_variance":
            max_var_criterion, max_var_stat_points = criterion, statistic_points
            mean_var_criterion, mean_var_stat_points = _make_criterion(MeanVarianceCriterion)
        else:
            mean_var_criterion, mean_var_stat_points = _make_criterion(MeanVarianceCriterion)
            max_var_criterion, max_var_stat_points = _make_criterion(MaxVarianceCriterion)

        # Collect all statistics in a single forward pass
        aggregator = StatisticsAggregatorFactory.create(wrapped_model, calibration_dataset)
        aggregator.stat_subset_size = subset_size
        for point_set in (statistic_points, mean_var_stat_points, max_var_stat_points):
            for point_list in point_set.values():
                for point in point_list:
                    aggregator.statistic_points.add_statistic_point(point)
        aggregator.collect_statistics(wrapped_model, graph)

    # Calculate primary scores (some metrics accept a dataset arg)
    import inspect
    calc_sig = inspect.signature(criterion._calc_sensitivity)
    if "dataset" in calc_sig.parameters:
        scores = criterion._calc_sensitivity(wrapped_model, graph, weight_params, statistic_points, calibration_dataset)
    else:
        scores = criterion._calc_sensitivity(wrapped_model, graph, weight_params, statistic_points)

    # Compute variance_ratios (max/mean per layer) for IQ/K sub-type assignment
    variance_ratios: dict[str, float] | None = None
    if mean_var_criterion is not None and max_var_criterion is not None:
        print("Computing variance ratios (max/mean) for IQ/K-quant assignment...")
        mean_scores = mean_var_criterion._calc_sensitivity(
            wrapped_model, graph, weight_params, mean_var_stat_points
        )
        max_scores = max_var_criterion._calc_sensitivity(
            wrapped_model, graph, weight_params, max_var_stat_points
        )
        variance_ratios = {
            weight_param.weight_name: max_score / (mean_score + _EPS)
            for weight_param, mean_score, max_score in zip(weight_params, mean_scores, max_scores)
        }
        median = sorted(variance_ratios.values())[len(variance_ratios) // 2]
        print(f"  Variance ratios computed for {len(variance_ratios)} layers (median={median:.2f})")

    # Build result
    layers = []
    for weight_param, score in zip(weight_params, scores):
        vr = variance_ratios.get(weight_param.weight_name) if variance_ratios else None
        layers.append(LayerSensitivity(
            layer_name=weight_param.weight_name,
            score=float(score),
            num_weights=int(weight_param.num_weights),
            variance_ratio=vr,
        ))

    print(f"Computed scores for {len(layers)} layers")
    return SensitivityResult(
        model_id=model_id, metric=metric, layers=layers,
    )


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
