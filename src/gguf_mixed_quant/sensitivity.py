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

    # Use NNCF's compress_weights with ratio=1.0 in a way that exposes scores.
    # We use the internal API to get scores without actually compressing.
    from nncf.quantization.algorithms.weight_compression.algorithm import WeightCompression
    from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig

    # Create the weight compression algorithm to extract scores
    compression_config = WeightCompressionConfig(
        mode=CompressWeightsMode.INT4_SYM,
        group_size=group_size,
    )

    wc_algo = WeightCompression(
        mode=CompressWeightsMode.INT4_SYM,
        ratio=0.8,  # ratio doesn't matter, we just want scores
        group_size=group_size,
        sensitivity_metric=sensitivity_metric,
        dataset=calibration_dataset,
    )

    # Get the graph and weight params
    from nncf.torch import wrap_model

    example_input = _get_example_input(model, tokenizer, device)
    wrapped_model = wrap_model(model, example_input=example_input, trace_parameters=True)
    graph = wrapped_model.nncf.get_graph()

    # Get weight nodes from the algorithm
    weight_params = wc_algo._get_weight_params(graph)

    # Compute sensitivity using the mixed precision criterion
    from nncf.quantization.algorithms.weight_compression.mixed_precision import MIXED_PRECISION_CRITERIA

    criterion_cls = MIXED_PRECISION_CRITERIA.get(sensitivity_metric)
    criterion = criterion_cls(ratio=0.8, subset_size=subset_size)

    # For data-aware, we need statistics
    statistic_points = None
    if metric not in DATA_FREE_METRICS:
        nodes_and_port_ids = [(wp.node_with_weight, wp.weight_port_id) for wp in weight_params]
        statistic_points = criterion.get_statistic_points(wrapped_model, graph, nodes_and_port_ids)

        # Collect statistics
        from nncf.common.tensor_statistics.aggregator import StatisticsAggregator

        aggregator = StatisticsAggregator(calibration_dataset)
        aggregator.register_statistic_points(statistic_points)
        aggregator.collect_statistics(wrapped_model, graph)

    # Calculate scores
    scores = criterion._calc_sensitivity(wrapped_model, graph, weight_params, statistic_points)

    # Build result
    layers = []
    for wp, score in zip(weight_params, scores):
        layer_name = wp.node_with_weight.node_name
        # Convert NNCF node name to HF-style layer name
        clean_name = _nncf_name_to_hf_name(layer_name)
        layers.append(LayerSensitivity(
            layer_name=clean_name,
            score=score,
            num_weights=wp.num_weights,
        ))

    print(f"Computed scores for {len(layers)} layers")
    return SensitivityResult(model_id=model_id, metric=metric, layers=layers)


def _get_example_input(model, tokenizer, device: str) -> dict:
    """Generate an example input for model tracing."""
    text = "Hello, this is a test."
    inputs = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def _nncf_name_to_hf_name(nncf_name: str) -> str:
    """
    Convert NNCF internal node name to HuggingFace-style weight name.

    NNCF names look like: '/model/layers.0/self_attn/q_proj/MatMul'
    HF names look like: 'model.layers.0.self_attn.q_proj.weight'
    """
    # Remove leading slash and trailing op name
    parts = nncf_name.strip("/").split("/")
    # Remove the last part if it's an operation name (MatMul, Add, etc.)
    if parts and parts[-1] in ("MatMul", "Add", "Multiply", "Linear"):
        parts = parts[:-1]

    # Join with dots and append .weight
    name = ".".join(parts)
    if not name.endswith(".weight"):
        name += ".weight"
    return name


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
