"""Map sensitivity scores to GGUF quantization types."""

from __future__ import annotations

from collections import Counter
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
    total_weights = sum(layer.num_weights for layer in sorted_layers)

    # Build cumulative thresholds from ratios
    # ratios are per-tier fractions: [0.75, 0.15, 0.10] → thresholds: [0.75, 0.90, 1.0]
    cumulative_thresholds = []
    running = 0.0
    for r in preset.ratios:
        running += r
        cumulative_thresholds.append(running)

    assignments = []
    accumulated_weights = 0

    for layer in sorted_layers:
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


# ---------------------------------------------------------------------------
# Reverse mapping: ggml type name (from llama-quantize output) → GGUFQuantType
# llama-quantize prints types like "q4_K", "q6_K", "q5_K", "q3_K", "q2_K",
# "q8_0", "f16", "f32", "iq4_xs", etc.
# We map to the canonical GGUFQuantType for each.
# ---------------------------------------------------------------------------
GGML_TO_QUANT_TYPE: dict[str, GGUFQuantType] = {
    "iq1_s": GGUFQuantType.IQ1_S,
    "iq1_m": GGUFQuantType.IQ1_M,
    "iq2_xxs": GGUFQuantType.IQ2_XXS,
    "iq2_xs": GGUFQuantType.IQ2_XS,
    "iq2_s": GGUFQuantType.IQ2_S,
    "q2_K": GGUFQuantType.Q2_K,
    "iq3_xxs": GGUFQuantType.IQ3_XXS,
    "iq3_s": GGUFQuantType.IQ3_S,
    "q3_K": GGUFQuantType.Q3_K_M,
    "iq4_xs": GGUFQuantType.IQ4_XS,
    "iq4_nl": GGUFQuantType.IQ4_NL,
    "q4_K": GGUFQuantType.Q4_K_M,
    "q4_0": GGUFQuantType.Q4_K_S,
    "q4_1": GGUFQuantType.Q4_K_S,
    "q5_K": GGUFQuantType.Q5_K_M,
    "q5_0": GGUFQuantType.Q5_K_S,
    "q5_1": GGUFQuantType.Q5_K_S,
    "q6_K": GGUFQuantType.Q6_K,
    "q8_0": GGUFQuantType.Q8_0,
    "f16": GGUFQuantType.F16,
}


# Step-down map for K-quant family: base type → one tier lower
_K_QUANT_STEP_DOWN: dict[GGUFQuantType, GGUFQuantType] = {
    GGUFQuantType.Q6_K: GGUFQuantType.Q5_K_M,
    GGUFQuantType.Q5_K_M: GGUFQuantType.Q4_K_M,
    GGUFQuantType.Q5_K_S: GGUFQuantType.Q4_K_S,
    GGUFQuantType.Q4_K_M: GGUFQuantType.Q3_K_M,
    GGUFQuantType.Q4_K_S: GGUFQuantType.Q3_K_S,
    GGUFQuantType.Q3_K_M: GGUFQuantType.Q2_K,
    GGUFQuantType.Q3_K_L: GGUFQuantType.Q3_K_M,
    GGUFQuantType.Q3_K_S: GGUFQuantType.Q2_K,
}


def _hf_to_gguf_name(hf_name: str) -> str:
    """Convert HuggingFace weight name to GGUF tensor name."""
    import re

    name = re.sub(r"model\.layers\.(\d+)", r"blk.\1", hf_name)
    replacements = {
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    for hf_suffix, gguf_suffix in replacements.items():
        if name.endswith(hf_suffix) or name == hf_suffix:
            prefix = name[: -len(hf_suffix)] if name.endswith(hf_suffix) else ""
            name = prefix + gguf_suffix
            break
    return name


def refine_baseline(
    baseline_map: dict[str, str],
    sensitivity_result: SensitivityResult,
    swap_count: int | None = None,
    dip_fraction: float = 0.0,
) -> MixedPrecisionPlan:
    """
    Refine llama.cpp's baseline by reranking type assignments by sensitivity.

    Takes the exact multiset of quantization types that llama.cpp assigned to
    scored layers, sorts them by BPW (ascending), sorts layers by sensitivity
    (ascending), and zips them together. Least-sensitive layers get the
    lowest-precision types, most-sensitive get the highest.

    This preserves the exact same total bits as the baseline (identical file
    size) while optimally distributing precision by sensitivity.

    :param baseline_map: {gguf_tensor_name: ggml_type} from llama-quantize.
    :param sensitivity_result: Output from compute_sensitivity().
    :param swap_count: Unused, kept for CLI compatibility.
    :param dip_fraction: Unused, kept for CLI compatibility.
    :return: MixedPrecisionPlan with reranked per-layer assignments.
    """
    sorted_layers = sensitivity_result.sorted_layers

    # Build HF→GGUF name map
    hf_to_gguf = {layer.layer_name: _hf_to_gguf_name(layer.layer_name) for layer in sorted_layers}

    # Convert baseline ggml types to GGUFQuantType
    baseline_types: dict[str, GGUFQuantType] = {}
    for gguf_name, ggml_type in baseline_map.items():
        qtype = GGML_TO_QUANT_TYPE.get(ggml_type)
        if qtype is not None:
            baseline_types[gguf_name] = qtype

    # Collect scored layers that have a baseline assignment
    matched_layers: list[LayerSensitivity] = []
    matched_types: list[GGUFQuantType] = []
    for layer in sorted_layers:
        gguf_name = hf_to_gguf[layer.layer_name]
        if gguf_name in baseline_types:
            matched_layers.append(layer)
            matched_types.append(baseline_types[gguf_name])

    if not matched_layers:
        raise ValueError("No scored layers matched baseline tensor names.")

    # Print baseline distribution
    baseline_dist = Counter(t.value for t in matched_types)
    print(f"  Baseline distribution (scored layers): {dict(baseline_dist)}")

    # Sort types by BPW ascending (lowest precision first)
    matched_types.sort(key=lambda t: get_bpw(t))

    # sorted_layers is already ascending by sensitivity (least sensitive first)
    # matched_layers preserves that order — zip: least sensitive → lowest BPW
    refined: dict[str, GGUFQuantType] = {}
    for layer, qtype in zip(matched_layers, matched_types):
        gguf_name = hf_to_gguf[layer.layer_name]
        refined[gguf_name] = qtype

    # Print refined distribution (should be identical multiset)
    refined_dist = Counter(t.value for t in refined.values())
    print(f"  Refined distribution (reranked):       {dict(refined_dist)}")

    # Build final assignments
    assignments = []
    for layer in sorted_layers:
        gguf_name = hf_to_gguf[layer.layer_name]
        quant_type = refined.get(gguf_name)
        if quant_type is None:
            # Unmatched layer — use most common baseline type as fallback
            quant_type = Counter(matched_types).most_common(1)[0][0]
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


def _baseline_to_plan(
    baseline_types: dict[str, GGUFQuantType],
    hf_to_gguf: dict[str, str],
    sorted_layers: list[LayerSensitivity],
    sensitivity_result: SensitivityResult,
    fallback_type: GGUFQuantType,
) -> MixedPrecisionPlan:
    """Convert baseline type map to a MixedPrecisionPlan."""
    assignments = []
    for layer in sorted_layers:
        gguf_name = hf_to_gguf[layer.layer_name]
        quant_type = baseline_types.get(gguf_name, fallback_type)
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


# ---------------------------------------------------------------------------
# Type ladder for Robin Hood mixed-family quantization.
# Ordered ascending by effective BPW.  Mixes IQ and K-quant families so
# the optimizer can freely trade bits across families — exactly like Unsloth
# Dynamic 2.0.
# ---------------------------------------------------------------------------
_TYPE_LADDER: list[GGUFQuantType] = [
    GGUFQuantType.IQ2_XXS,   # 2.06 bpw
    GGUFQuantType.IQ2_XS,    # 2.31 bpw
    GGUFQuantType.Q2_K,      # 2.63 bpw
    GGUFQuantType.IQ3_XXS,   # 3.06 bpw
    GGUFQuantType.IQ3_S,     # 3.44 bpw
    GGUFQuantType.Q3_K_S,    # 3.44 bpw
    GGUFQuantType.Q3_K_M,    # 3.91 bpw
    GGUFQuantType.IQ4_XS,    # 4.25 bpw
    GGUFQuantType.Q4_K_S,    # 4.59 bpw
    GGUFQuantType.Q4_K_M,    # 4.85 bpw
    GGUFQuantType.Q5_K_S,    # 5.54 bpw
    GGUFQuantType.Q5_K_M,    # 5.69 bpw
    GGUFQuantType.Q6_K,      # 6.56 bpw
    GGUFQuantType.Q8_0,      # 8.50 bpw
]


def _snap_to_ladder(
    target_bpw: float,
    ladder: list[GGUFQuantType],
    ladder_bpw: dict[GGUFQuantType, float],
) -> int:
    """Return ladder index of the type whose BPW is closest to target_bpw."""
    best_idx = 0
    best_dist = abs(ladder_bpw[ladder[0]] - target_bpw)
    for i, t in enumerate(ladder):
        dist = abs(ladder_bpw[t] - target_bpw)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def robin_hood(
    baseline_map: dict[str, str],
    sensitivity_result: SensitivityResult,
    extra_bpw: float = 0.0,
) -> MixedPrecisionPlan:
    """
    Robin Hood mixed-family quantization: steal bits from insensitive
    layers, give them to sensitive layers.

    Uses normalized sensitivity scores to determine the quantization type
    for each layer.  The score is mapped to a position on the type ladder:
    low-sensitivity layers get IQ4_XS (cross-family downgrade), the bulk
    stays near the baseline's primary type, and high-sensitivity layers
    get proportionally higher types (Q5_K, Q6_K).

    The total bit budget is matched to the baseline by iteratively
    adjusting a threshold parameter.

    :param baseline_map: {gguf_tensor_name: ggml_type} from llama-quantize.
    :param sensitivity_result: Output from compute_sensitivity().
    :param extra_bpw: Extra average bits-per-weight budget above baseline
        (0.0 = same total size, positive = allow a larger file).
    :return: MixedPrecisionPlan with mixed-family per-layer assignments.
    """
    sorted_layers = sensitivity_result.sorted_layers

    # Build HF→GGUF name map
    hf_to_gguf = {
        layer.layer_name: _hf_to_gguf_name(layer.layer_name)
        for layer in sorted_layers
    }

    # Convert baseline ggml types to GGUFQuantType
    baseline_types: dict[str, GGUFQuantType] = {}
    for gguf_name, ggml_type in baseline_map.items():
        qtype = GGML_TO_QUANT_TYPE.get(ggml_type)
        if qtype is not None:
            baseline_types[gguf_name] = qtype

    # Match scored layers to baseline assignments
    matched: list[tuple[LayerSensitivity, GGUFQuantType]] = []
    for layer in sorted_layers:
        gguf_name = hf_to_gguf[layer.layer_name]
        if gguf_name in baseline_types:
            matched.append((layer, baseline_types[gguf_name]))

    if not matched:
        raise ValueError("No scored layers matched baseline tensor names.")

    # Compute baseline bit budget
    baseline_bits = sum(
        get_bpw(qtype) * layer.num_weights for layer, qtype in matched
    )
    total_weights = sum(layer.num_weights for layer, _ in matched)
    budget_bits = baseline_bits + extra_bpw * total_weights

    baseline_avg_bpw = baseline_bits / total_weights
    budget_avg_bpw = budget_bits / total_weights

    baseline_dist = Counter(qt.value for _, qt in matched)
    print(f"  Baseline distribution: {dict(baseline_dist)}")
    print(f"  Baseline avg BPW:     {baseline_avg_bpw:.3f}")
    print(f"  Budget avg BPW:       {budget_avg_bpw:.3f}")

    # Identify baseline primary type (most common)
    primary_type = Counter(qt for _, qt in matched).most_common(1)[0][0]
    primary_bpw = get_bpw(primary_type)

    # Build the working ladder: IQ4_XS + primary + upgrade tiers.
    # Use a clean 4-type ladder like Unsloth: IQ4_XS, Q4_K, Q5_K, Q6_K.
    # Wider BPW gaps between types produce a more spread distribution.
    down_type = GGUFQuantType.IQ4_XS
    # Pick distinct upgrade tiers with meaningful BPW gaps (>= 0.5 bpw apart)
    candidates = [
        t for t in _TYPE_LADDER
        if primary_bpw < get_bpw(t) <= primary_bpw + 2.0
    ]
    upgrade_types: list[GGUFQuantType] = []
    last_bpw = primary_bpw
    for t in candidates:
        if get_bpw(t) >= last_bpw + 0.5:
            upgrade_types.append(t)
            last_bpw = get_bpw(t)
    if not upgrade_types:
        upgrade_types = candidates[:2] if candidates else [_TYPE_LADDER[-1]]

    ladder = [down_type, primary_type] + upgrade_types
    ladder_bpw = [get_bpw(t) for t in ladder]
    num_levels = len(ladder)

    print(f"  Ladder: {[f'{t.value}({b:.2f})' for t, b in zip(ladder, ladder_bpw)]}")

    # Layers are already sorted ascending by sensitivity score from
    # sorted_layers.  The budget-adjustment step naturally weights by
    # num_weights when computing bit costs, so explicit param weighting
    # in the ranking is unnecessary.
    layers_by_sensitivity = [layer for layer, _ in matched]
    n = len(layers_by_sensitivity)

    # Use RANK-based mapping instead of score-based.
    # Sensitivity scores are extremely right-skewed (often 1000x between
    # min and max), so linear normalization piles everything into one bin.
    # Rank position gives a uniform [0, 1] distribution.
    rank_norm = [i / max(n - 1, 1) for i in range(n)]

    # Map normalized score → ladder level.
    #
    # Strategy: fix the bottom ~18% as IQ4_XS (like Unsloth), then
    # distribute the entire budget (baseline + IQ4_XS savings) across
    # the remaining layers proportionally to their sensitivity score.
    #
    # The sensitivity score determines the upgrade level:
    #   - bottom ~18% → IQ4_XS (index 0), regardless of score
    #   - remaining layers → mapped proportionally across [primary .. Q6_K]
    #     based on their normalized score among the non-downgraded set
    #
    # Binary search for the fraction of IQ4_XS layers that lets us spend
    # all the budget on upgrades without going over.

    def assign_with_iq_fraction(iq_frac: float) -> list[int]:
        """Assign IQ4_XS to bottom iq_frac layers, distribute rest by rank."""
        iq_count = int(iq_frac * n)
        upper_n = n - iq_count
        indices = []

        for rank in range(n):
            if rank < iq_count:
                indices.append(0)  # IQ4_XS
            else:
                # Rank within the non-IQ portion: 0 → upper_n-1
                upper_rank = rank - iq_count
                # Map uniformly to [1 .. num_levels-1]
                frac = upper_rank / max(upper_n - 1, 1)
                level = 1 + int(frac * (num_levels - 2) + 0.5)
                level = max(1, min(num_levels - 1, level))
                indices.append(level)
        return indices

    def bits_for_assignment(indices: list[int]) -> float:
        return sum(
            ladder_bpw[indices[i]] * layers_by_sensitivity[i].num_weights
            for i in range(n)
        )

    # Binary search for IQ fraction that matches budget.
    # Enforce minimum ~18% IQ4_XS (like Unsloth) — the savings from
    # IQ4_XS fund upgrades that improve PPL on sensitive layers.
    min_iq_frac = 0.18
    lo_frac, hi_frac = min_iq_frac, 0.20  # 18-20% IQ4_XS (like Unsloth's ~18%)
    for _ in range(60):
        mid = (lo_frac + hi_frac) / 2.0
        indices = assign_with_iq_fraction(mid)
        used = bits_for_assignment(indices)
        if used > budget_bits:
            # Over budget → need more IQ4_XS to save bits
            lo_frac = mid
        else:
            hi_frac = mid

    # Use fraction that stays within budget
    best_frac = lo_frac
    best_indices = assign_with_iq_fraction(best_frac)
    iq_count = int(best_frac * n)

    # Spend remaining bits: upgrade most-sensitive layers first
    used_bits = bits_for_assignment(best_indices)
    for i in range(n - 1, -1, -1):
        if used_bits >= budget_bits:
            break
        if best_indices[i] < num_levels - 1:
            w = layers_by_sensitivity[i].num_weights
            next_cost = (ladder_bpw[best_indices[i] + 1] - ladder_bpw[best_indices[i]]) * w
            if used_bits + next_cost <= budget_bits:
                used_bits += next_cost
                best_indices[i] += 1

    print(f"  IQ4_XS fraction: {best_frac:.1%} ({iq_count} layers)")

    # Build the refined map
    refined: dict[str, GGUFQuantType] = {}
    for i, layer in enumerate(layers_by_sensitivity):
        gguf_name = hf_to_gguf[layer.layer_name]
        refined[gguf_name] = ladder[best_indices[i]]

    refined_dist = Counter(t.value for t in refined.values())
    refined_bits = sum(
        get_bpw(refined[hf_to_gguf[layer.layer_name]]) * layer.num_weights
        for layer in layers_by_sensitivity
    )
    print(f"  Robin Hood distribution: {dict(refined_dist)}")
    print(f"  Robin Hood avg BPW:      {refined_bits / total_weights:.3f}")

    # Build final assignment list
    assignments = []
    for layer in sorted_layers:
        gguf_name = hf_to_gguf[layer.layer_name]
        quant_type = refined.get(gguf_name)
        if quant_type is None:
            quant_type = primary_type
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
