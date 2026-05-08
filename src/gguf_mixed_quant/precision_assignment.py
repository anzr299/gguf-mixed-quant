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
    Refine llama.cpp's baseline assignments using sensitivity scores.

    Keeps the same total bit budget as the baseline but redistributes the
    multi-tier type assignments to layers ranked by sensitivity. The most
    sensitive layers get the highest-precision types, progressively stepping
    down. This preserves total model size while directing precision where
    it matters most.

    When dip_fraction > 0, additionally downgrades the least-sensitive
    base-type layers to one tier below, and uses the saved bits to promote
    more sensitive base-type layers upward. This creates extra headroom
    for quality-critical layers at the expense of insensitive ones.

    :param baseline_map: {gguf_tensor_name: ggml_type} from llama-quantize.
    :param sensitivity_result: Output from compute_sensitivity().
    :param swap_count: Ignored (kept for CLI compat). Budget is fully redistributed.
    :param dip_fraction: Fraction of base-type weights to downgrade (0.0–1.0).
    :return: MixedPrecisionPlan with refined per-layer assignments.
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

    # Identify the base type (most common among scored layers)
    scored_types: list[GGUFQuantType] = []
    for layer in sorted_layers:
        gguf_name = hf_to_gguf[layer.layer_name]
        if gguf_name in baseline_types:
            scored_types.append(baseline_types[gguf_name])

    if not scored_types:
        raise ValueError("No scored layers matched baseline tensor names.")

    type_counts = Counter(scored_types)
    base_type = type_counts.most_common(1)[0][0]
    base_bpw = get_bpw(base_type)

    # Collect all bump tiers with their bit budgets (scored layers only)
    # Each tier = (GGUFQuantType, total_bits_above_base for that tier in baseline)
    bump_tiers: list[tuple[GGUFQuantType, float]] = []
    for qtype, count in type_counts.items():
        if get_bpw(qtype) <= base_bpw:
            continue
        # Sum the actual extra bits this tier uses in the baseline
        tier_budget = 0.0
        for layer in sorted_layers:
            gguf_name = hf_to_gguf[layer.layer_name]
            if gguf_name in baseline_types and baseline_types[gguf_name] == qtype:
                tier_budget += (get_bpw(qtype) - base_bpw) * layer.num_weights
        bump_tiers.append((qtype, tier_budget))

    # Check if the highest bump type in the full baseline is used by
    # embedding/lm_head (non-scored tensors). If the highest tier is used
    # ONLY by those special layers (not by any scored layers), skip it.
    # If scored layers also share that tier, keep it for redistribution.
    _EMBED_NAMES = {"token_embd.weight", "output.weight"}
    all_baseline_types = set(baseline_types.values())
    all_bump_types_full = sorted(
        [t for t in all_baseline_types if get_bpw(t) > base_bpw],
        key=lambda t: get_bpw(t), reverse=True,
    )
    if all_bump_types_full:
        highest_type = all_bump_types_full[0]
        embed_uses_highest = any(
            baseline_types.get(name) == highest_type for name in _EMBED_NAMES
        )
        # Check if any scored layer also uses this type
        scored_uses_highest = any(t == highest_type for t in scored_types)
        if embed_uses_highest and not scored_uses_highest:
            # Only embeddings use this tier — skip it entirely
            bump_tiers = [(t, b) for t, b in bump_tiers if t != highest_type]
            print(f"  Skipping {highest_type.value} tier (used only by embedding/lm_head)")
        elif embed_uses_highest and scored_uses_highest:
            print(f"  Note: {highest_type.value} tier shared by embedding + scored layers, redistributing scored portion")

    if not bump_tiers:
        print("  No bumped layers in baseline — nothing to refine.")
        return _baseline_to_plan(baseline_types, hf_to_gguf, sorted_layers,
                                 sensitivity_result, base_type)

    # Sort tiers by BPW descending (highest precision first)
    bump_tiers.sort(key=lambda x: get_bpw(x[0]), reverse=True)

    # Sort layers descending by sensitivity (most sensitive first)
    layers_desc = list(reversed(sorted_layers))

    # Greedily assign tiers: highest-BPW tier to the most-sensitive layers
    refined: dict[str, GGUFQuantType] = {}
    tier_counts: dict[str, int] = {}
    assigned = set()

    for tier_type, tier_budget in bump_tiers:
        tier_bpw = get_bpw(tier_type)
        remaining = tier_budget
        count = 0

        for layer in layers_desc:
            if layer.layer_name in assigned:
                continue
            gguf_name = hf_to_gguf[layer.layer_name]
            if gguf_name not in baseline_types:
                continue

            cost = (tier_bpw - base_bpw) * layer.num_weights
            if remaining >= cost:
                refined[gguf_name] = tier_type
                remaining -= cost
                assigned.add(layer.layer_name)
                count += 1

        tier_counts[tier_type.value] = count

    # Everything else stays at base
    for layer in sorted_layers:
        gguf_name = hf_to_gguf[layer.layer_name]
        if gguf_name not in refined and gguf_name in baseline_types:
            refined[gguf_name] = base_type

    # ------------------------------------------------------------------
    # Phase 2: Downgrade insensitive layers, promote sensitive base layers
    # ------------------------------------------------------------------
    dip_count = 0
    promote_count = 0
    dip_type = _K_QUANT_STEP_DOWN.get(base_type)

    if dip_fraction > 0 and dip_type is not None:
        dip_bpw = get_bpw(dip_type)
        # Lowest bump tier is used for promotion (cheapest upgrade)
        promote_type = bump_tiers[-1][0] if bump_tiers else None
        promote_bpw = get_bpw(promote_type) if promote_type else 0.0

        # Collect base-type layers (not already bumped), ascending sensitivity
        base_layers = [
            layer for layer in sorted_layers
            if refined.get(hf_to_gguf[layer.layer_name]) == base_type
        ]

        total_base_weights = sum(l.num_weights for l in base_layers)
        dip_budget_weights = total_base_weights * dip_fraction

        # Downgrade least-sensitive base layers (sorted ascending = least first)
        dip_bits_saved = 0.0
        dipped_names: set[str] = set()
        acc_weights = 0
        for layer in base_layers:
            if acc_weights + layer.num_weights > dip_budget_weights:
                break
            gguf_name = hf_to_gguf[layer.layer_name]
            refined[gguf_name] = dip_type
            dip_bits_saved += (base_bpw - dip_bpw) * layer.num_weights
            dipped_names.add(layer.layer_name)
            acc_weights += layer.num_weights
            dip_count += 1

        # Promote most-sensitive remaining base layers using saved bits
        if promote_type and dip_bits_saved > 0:
            remaining_budget = dip_bits_saved
            for layer in reversed(base_layers):
                if layer.layer_name in dipped_names:
                    continue
                gguf_name = hf_to_gguf[layer.layer_name]
                if refined.get(gguf_name) != base_type:
                    continue
                cost = (promote_bpw - base_bpw) * layer.num_weights
                if remaining_budget >= cost:
                    refined[gguf_name] = promote_type
                    remaining_budget -= cost
                    promote_count += 1

    # Print summary
    baseline_dist = {t.value: c for t, c in type_counts.items() if get_bpw(t) > base_bpw}
    print(f"  Baseline base type: {base_type.value}")
    print(f"  Baseline bump distribution: {baseline_dist}")
    print(f"  Refined bump distribution:  {tier_counts}")
    if dip_count > 0:
        print(f"  Dip phase: {dip_count} layers → {dip_type.value}, "
              f"{promote_count} layers promoted → {bump_tiers[-1][0].value if bump_tiers else '?'}")

    # Build final assignments
    assignments = []
    for layer in sorted_layers:
        gguf_name = hf_to_gguf[layer.layer_name]
        quant_type = refined.get(gguf_name, base_type)
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
