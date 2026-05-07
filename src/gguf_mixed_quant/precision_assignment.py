"""Map sensitivity scores to GGUF quantization types."""

from dataclasses import dataclass

from gguf_mixed_quant.gguf_types import (
    GGUF_QUANT_INFO,
    GGUF_TYPES_BY_PRECISION,
    GGUFQuantType,
    get_bpw,
    parse_quant_type,
)
from gguf_mixed_quant.sensitivity import LayerSensitivity, SensitivityResult


@dataclass
class LayerAssignment:
    """Quantization type assignment for a single layer."""

    layer_name: str
    quant_type: GGUFQuantType
    score: float
    num_weights: int

    @property
    def bits_per_weight(self) -> float:
        return get_bpw(self.quant_type)


@dataclass
class MixedPrecisionPlan:
    """Complete mixed-precision quantization plan."""

    model_id: str
    metric: str
    assignments: list[LayerAssignment]

    @property
    def avg_bpw(self) -> float:
        """Average bits-per-weight across all layers, weighted by num_weights."""
        total_bits = sum(a.bits_per_weight * a.num_weights for a in self.assignments)
        total_weights = sum(a.num_weights for a in self.assignments)
        if total_weights == 0:
            return 0.0
        return total_bits / total_weights

    @property
    def type_distribution(self) -> dict[str, int]:
        """Count of layers per quantization type."""
        dist: dict[str, int] = {}
        for a in self.assignments:
            key = a.quant_type.value
            dist[key] = dist.get(key, 0) + 1
        return dist

    def summary(self) -> str:
        """Human-readable summary of the plan."""
        lines = [
            f"Model: {self.model_id}",
            f"Metric: {self.metric}",
            f"Total layers: {len(self.assignments)}",
            f"Average BPW: {self.avg_bpw:.2f}",
            "Distribution:",
        ]
        for qtype, count in sorted(self.type_distribution.items()):
            lines.append(f"  {qtype}: {count} layers")
        return "\n".join(lines)


def assign_gguf_types(
    sensitivity_result: SensitivityResult,
    ratio: float = 0.8,
    primary_type: str = "Q4_K_M",
    backup_type: str = "Q6_K",
) -> MixedPrecisionPlan:
    """
    Assign GGUF quantization types using a two-level scheme (like NNCF's ratio-based approach).

    Layers are sorted by sensitivity. The least-sensitive layers (up to `ratio` fraction
    of total weights) get `primary_type`, the rest get `backup_type`.

    :param sensitivity_result: Output from compute_sensitivity().
    :param ratio: Fraction of weights to assign to primary (lower) precision.
    :param primary_type: GGUF type for least-sensitive layers.
    :param backup_type: GGUF type for most-sensitive layers.
    :return: MixedPrecisionPlan with per-layer assignments.
    """
    primary = parse_quant_type(primary_type)
    backup = parse_quant_type(backup_type)

    sorted_layers = sensitivity_result.sorted_layers
    total_weights = sum(layer.num_weights for layer in sorted_layers)

    assignments = []
    accumulated_weights = 0

    for layer in sorted_layers:
        current_ratio = (accumulated_weights + layer.num_weights) / total_weights
        if current_ratio <= ratio:
            quant_type = primary
            accumulated_weights += layer.num_weights
        else:
            quant_type = backup

        assignments.append(LayerAssignment(
            layer_name=layer.layer_name,
            quant_type=quant_type,
            score=layer.score,
            num_weights=layer.num_weights,
        ))

    return MixedPrecisionPlan(
        model_id=sensitivity_result.model_id,
        metric=sensitivity_result.metric,
        assignments=assignments,
    )


def assign_gguf_types_multilevel(
    sensitivity_result: SensitivityResult,
    num_levels: int = 4,
    quant_types: list[str] | None = None,
) -> MixedPrecisionPlan:
    """
    Assign GGUF quantization types using multiple precision levels.

    Layers are sorted by sensitivity and divided into `num_levels` equal-sized buckets.
    Each bucket gets progressively higher precision.

    :param sensitivity_result: Output from compute_sensitivity().
    :param num_levels: Number of distinct quantization levels.
    :param quant_types: Explicit list of GGUF types from lowest to highest precision.
        If None, automatically selects types spread across the available range.
    :return: MixedPrecisionPlan with per-layer assignments.
    """
    if quant_types is not None:
        types = [parse_quant_type(t) for t in quant_types]
        if len(types) != num_levels:
            raise ValueError(f"Expected {num_levels} quant types, got {len(types)}")
    else:
        types = _select_spread_types(num_levels)

    sorted_layers = sensitivity_result.sorted_layers
    n = len(sorted_layers)
    bucket_size = n // num_levels

    assignments = []
    for i, layer in enumerate(sorted_layers):
        # Determine which bucket this layer falls into
        bucket_idx = min(i // bucket_size, num_levels - 1) if bucket_size > 0 else 0
        quant_type = types[bucket_idx]

        assignments.append(LayerAssignment(
            layer_name=layer.layer_name,
            quant_type=quant_type,
            score=layer.score,
            num_weights=layer.num_weights,
        ))

    return MixedPrecisionPlan(
        model_id=sensitivity_result.model_id,
        metric=sensitivity_result.metric,
        assignments=assignments,
    )


def _select_spread_types(num_levels: int) -> list[GGUFQuantType]:
    """Select quantization types evenly spread across the precision range."""
    available = GGUF_TYPES_BY_PRECISION
    if num_levels >= len(available):
        return list(available)

    # Spread evenly
    step = (len(available) - 1) / (num_levels - 1) if num_levels > 1 else 0
    indices = [round(i * step) for i in range(num_levels)]
    return [available[i] for i in indices]
