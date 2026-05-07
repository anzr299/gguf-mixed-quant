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


def _transform_fn(data: dict, tokenizer) -> dict:
    """Tokenize text data for calibration."""
    tokenized = tokenizer(data["text"], return_tensors="pt")
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }


def _build_calibration_dataset(
    dataset_name: str,
    tokenizer,
    subset_size: int = 128,
) -> nncf.Dataset:
    """Build a calibration dataset from HuggingFace datasets."""
    from datasets import load_dataset

    if dataset_name in ("wikitext", "wikitext-2"):
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    else:
        ds = load_dataset(dataset_name, split="train")

    # Filter empty texts
    ds = ds.filter(lambda x: len(x["text"].strip()) > 10)
    ds = ds.select(range(min(subset_size, len(ds))))

    return nncf.Dataset(ds, partial(_transform_fn, tokenizer=tokenizer))


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
        nodes_and_port_ids = [(wp.node_with_weight, wp.weight_port_id) for wp in weight_params]
        statistic_points = criterion.get_statistic_points(wrapped_model, graph, nodes_and_port_ids)

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
