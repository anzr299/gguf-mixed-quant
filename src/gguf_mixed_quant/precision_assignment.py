"""Map sensitivity scores to GGUF quantization types."""

from dataclasses import dataclass, field

from gguf_mixed_quant.gguf_types import (
    GGUF_QUANT_INFO,
    GGUF_TYPES_BY_PRECISION,
    GGUFQuantType,
    get_bpw,
    parse_quant_type,
)
from gguf_mixed_quant.sensitivity import LayerSensitivity, SensitivityResult


# ---------------------------------------------------------------------------
# Multi-level presets matching llama.cpp's quantization presets.
# Each preset defines the same tier types that llama.cpp uses internally
# (base type + bump-up types for sensitive layers), but the assignment is
# driven by sensitivity scores rather than fixed layer-position rules.
#
# From llama-quant.cpp's use_more_bits(): ~25-35% of layers get bumped.
# Attention_v and ffn_down first/last 1/8 are the typical bump targets.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantPreset:
    """A named quantization preset with multiple precision tiers.

    Tiers are ordered from lowest to highest precision.
    Ratios define the cumulative weight fraction for each tier (must sum to 1.0).
    """

    name: str
    description: str
    tiers: list[GGUFQuantType]
    ratios: list[float]

    def __post_init__(self) -> None:
        if len(self.tiers) != len(self.ratios):
            raise ValueError("tiers and ratios must have the same length")
        if abs(sum(self.ratios) - 1.0) > 1e-6:
            raise ValueError(f"ratios must sum to 1.0, got {sum(self.ratios):.4f}")


# Presets mirror llama.cpp's ftype distribution logic:
# - "base" type covers the bulk of layers
# - sensitive layers (detected by score) get bumped to higher types
# - most critical layers (output, attention_v) get the highest type
#
# Ratios derived from llama.cpp's use_more_bits (first/last 1/8 + every 3rd = ~30%)
# and explicit per-category bumps.

PRESETS: dict[str, QuantPreset] = {
    # --- IQ (importance) quants ---
    "IQ2_XXS": QuantPreset(
        name="IQ2_XXS",
        description="~2-bit importance quants, critical layers bumped to Q4_K/Q5_K",
        tiers=[GGUFQuantType.IQ2_XXS, GGUFQuantType.Q2_K, GGUFQuantType.Q5_K_S],
        ratios=[0.70, 0.20, 0.10],
    ),
    "IQ2_XS": QuantPreset(
        name="IQ2_XS",
        description="~2.3-bit importance quants, sensitive layers get Q4_K",
        tiers=[GGUFQuantType.IQ2_XS, GGUFQuantType.Q4_K_S, GGUFQuantType.Q5_K_S],
        ratios=[0.70, 0.20, 0.10],
    ),
    "IQ2_S": QuantPreset(
        name="IQ2_S",
        description="~2.5-bit importance quants, bumps to IQ3_S/Q4_K",
        tiers=[GGUFQuantType.IQ2_XS, GGUFQuantType.IQ3_S, GGUFQuantType.Q4_K_S],
        ratios=[0.65, 0.25, 0.10],
    ),
    "IQ2_M": QuantPreset(
        name="IQ2_M",
        description="~2.7-bit importance quants, bumps to IQ3_S/Q4_K",
        tiers=[GGUFQuantType.IQ2_S, GGUFQuantType.IQ3_S, GGUFQuantType.Q4_K_S],
        ratios=[0.65, 0.25, 0.10],
    ),
    "IQ3_XXS": QuantPreset(
        name="IQ3_XXS",
        description="~3-bit importance quants, sensitive → Q4_K",
        tiers=[GGUFQuantType.IQ3_XXS, GGUFQuantType.Q4_K_S, GGUFQuantType.Q5_K_S],
        ratios=[0.70, 0.20, 0.10],
    ),
    "IQ3_XS": QuantPreset(
        name="IQ3_XS",
        description="~3.3-bit importance quants, bumps to Q4_K",
        tiers=[GGUFQuantType.IQ3_S, GGUFQuantType.Q4_K_S, GGUFQuantType.Q5_K_S],
        ratios=[0.70, 0.20, 0.10],
    ),
    "IQ3_S": QuantPreset(
        name="IQ3_S",
        description="~3.4-bit importance quants, GQA-sensitive → Q4_K",
        tiers=[GGUFQuantType.IQ3_S, GGUFQuantType.Q4_K_S, GGUFQuantType.Q5_K_S],
        ratios=[0.72, 0.18, 0.10],
    ),
    "IQ3_M": QuantPreset(
        name="IQ3_M",
        description="~3.7-bit importance quants, sensitive ffn_down → Q4_K",
        tiers=[GGUFQuantType.IQ3_S, GGUFQuantType.Q4_K_M, GGUFQuantType.Q5_K_S],
        ratios=[0.68, 0.22, 0.10],
    ),
    "IQ4_XS": QuantPreset(
        name="IQ4_XS",
        description="~4.25-bit importance quants, critical → Q5_K",
        tiers=[GGUFQuantType.IQ4_XS, GGUFQuantType.Q5_K_S, GGUFQuantType.Q6_K],
        ratios=[0.75, 0.15, 0.10],
    ),
    "IQ4_NL": QuantPreset(
        name="IQ4_NL",
        description="~4.5-bit importance quants (non-linear), critical → Q5_K",
        tiers=[GGUFQuantType.IQ4_NL, GGUFQuantType.Q5_K_S, GGUFQuantType.Q6_K],
        ratios=[0.75, 0.15, 0.10],
    ),
    # --- K-quants ---
    "Q2_K": QuantPreset(
        name="Q2_K",
        description="2-bit K-quants, attn_v→Q3_K/Q4_K, ffn_down→Q3_K, output→Q6_K",
        tiers=[GGUFQuantType.Q2_K, GGUFQuantType.Q3_K_M, GGUFQuantType.Q4_K_S, GGUFQuantType.Q6_K],
        ratios=[0.55, 0.25, 0.12, 0.08],
    ),
    "Q2_K_S": QuantPreset(
        name="Q2_K_S",
        description="2-bit K-quants (small), minimal bumps",
        tiers=[GGUFQuantType.Q2_K, GGUFQuantType.Q4_K_S, GGUFQuantType.Q6_K],
        ratios=[0.78, 0.15, 0.07],
    ),
    "Q3_K_S": QuantPreset(
        name="Q3_K_S",
        description="3-bit K-quants (small), output→Q6_K only",
        tiers=[GGUFQuantType.Q3_K_S, GGUFQuantType.Q5_K_S, GGUFQuantType.Q6_K],
        ratios=[0.82, 0.12, 0.06],
    ),
    "Q3_K_M": QuantPreset(
        name="Q3_K_M",
        description="3-bit K-quants (medium), attn_v→Q4_K/Q5_K, ffn_down→Q4_K/Q5_K",
        tiers=[GGUFQuantType.Q3_K_M, GGUFQuantType.Q4_K_M, GGUFQuantType.Q5_K_M, GGUFQuantType.Q6_K],
        ratios=[0.55, 0.25, 0.13, 0.07],
    ),
    "Q3_K_L": QuantPreset(
        name="Q3_K_L",
        description="3-bit K-quants (large), attn_v→Q5_K, ffn_down→Q5_K",
        tiers=[GGUFQuantType.Q3_K_L, GGUFQuantType.Q4_K_M, GGUFQuantType.Q5_K_M, GGUFQuantType.Q6_K],
        ratios=[0.50, 0.25, 0.17, 0.08],
    ),
    "Q4_K_S": QuantPreset(
        name="Q4_K_S",
        description="4-bit K-quants (small), first few attn_v/ffn_down→Q5_K",
        tiers=[GGUFQuantType.Q4_K_S, GGUFQuantType.Q5_K_S, GGUFQuantType.Q6_K],
        ratios=[0.78, 0.15, 0.07],
    ),
    "Q4_K_M": QuantPreset(
        name="Q4_K_M",
        description="4-bit K-quants (medium), ~30% sensitive→Q6_K via use_more_bits",
        tiers=[GGUFQuantType.Q4_K_M, GGUFQuantType.Q5_K_M, GGUFQuantType.Q6_K],
        ratios=[0.65, 0.20, 0.15],
    ),
    "Q5_K_S": QuantPreset(
        name="Q5_K_S",
        description="5-bit K-quants (small), output→Q6_K",
        tiers=[GGUFQuantType.Q5_K_S, GGUFQuantType.Q6_K],
        ratios=[0.88, 0.12],
    ),
    "Q5_K_M": QuantPreset(
        name="Q5_K_M",
        description="5-bit K-quants (medium), ~30% sensitive→Q6_K via use_more_bits",
        tiers=[GGUFQuantType.Q5_K_M, GGUFQuantType.Q6_K],
        ratios=[0.70, 0.30],
    ),
    "Q6_K": QuantPreset(
        name="Q6_K",
        description="6-bit K-quants, output→Q8_0",
        tiers=[GGUFQuantType.Q6_K, GGUFQuantType.Q8_0],
        ratios=[0.92, 0.08],
    ),
    "Q8_0": QuantPreset(
        name="Q8_0",
        description="8-bit, output→F16",
        tiers=[GGUFQuantType.Q8_0, GGUFQuantType.F16],
        ratios=[0.95, 0.05],
    ),
}


def list_presets() -> dict[str, str]:
    """Return available preset names and descriptions."""
    return {name: preset.description for name, preset in PRESETS.items()}


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


def assign_gguf_types_preset(
    sensitivity_result: SensitivityResult,
    preset_name: str = "Q4_K_M",
) -> MixedPrecisionPlan:
    """
    Assign GGUF types using a named preset that mimics llama.cpp's multi-level logic.

    Layers are sorted by sensitivity score. The preset's tier ratios determine what
    fraction of total weights goes into each precision tier. Most-sensitive layers
    get the highest precision tier.

    :param sensitivity_result: Output from compute_sensitivity().
    :param preset_name: Name of the preset (see PRESETS dict or list_presets()).
    :return: MixedPrecisionPlan with per-layer assignments.
    """
    preset_key = preset_name.upper().replace("-", "_")
    if preset_key not in PRESETS:
        valid = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset: '{preset_name}'. Available: {valid}")

    preset = PRESETS[preset_key]

    # Sort layers ascending by score (least sensitive first)
    sorted_layers = sensitivity_result.sorted_layers
    # Exclude embedding/output (inf score) from ratio calculation
    total_weights = sum(layer.num_weights for layer in sorted_layers if layer.score != float("inf"))

    # Build cumulative thresholds from ratios
    # ratios are per-tier fractions: [0.75, 0.15, 0.10] → thresholds: [0.75, 0.90, 1.0]
    cumulative_thresholds = []
    running = 0.0
    for r in preset.ratios:
        running += r
        cumulative_thresholds.append(running)

    # Embedding/output layers (score=inf) always get Q6_K, matching llama.cpp default
    EMBEDDING_OUTPUT_TYPE = GGUFQuantType.Q6_K

    assignments = []
    accumulated_weights = 0

    for layer in sorted_layers:
        # Force embedding/output layers to Q6_K regardless of preset
        if layer.score == float("inf"):
            assignments.append(LayerAssignment(
                layer_name=layer.layer_name,
                quant_type=EMBEDDING_OUTPUT_TYPE,
                score=layer.score,
                num_weights=layer.num_weights,
            ))
            continue

        weight_ratio = (accumulated_weights + layer.num_weights) / total_weights
        accumulated_weights += layer.num_weights

        # Find which tier this layer falls into
        tier_idx = 0
        for i, threshold in enumerate(cumulative_thresholds):
            if weight_ratio <= threshold + 1e-9:
                tier_idx = i
                break
        else:
            tier_idx = len(preset.tiers) - 1

        assignments.append(LayerAssignment(
            layer_name=layer.layer_name,
            quant_type=preset.tiers[tier_idx],
            score=layer.score,
            num_weights=layer.num_weights,
        ))

    return MixedPrecisionPlan(
        model_id=sensitivity_result.model_id,
        metric=sensitivity_result.metric,
        assignments=assignments,
    )
