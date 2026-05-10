"""Map sensitivity scores to GGUF quantization types.

Two assignment paths:
  1. Auto mode (two_phase_assign): sensitivity-ranked band allocation with
     adaptive band sizing, IQ budget control, and variance-ratio-driven
     sub-type selection.
  2. Manual mode (assign_gguf_types_preset): user-specified tiers and ratios.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from gguf_mixed_quant.gguf_types import (
    GGUFQuantType,
    get_bpw,
)
from gguf_mixed_quant.sensitivity import LayerSensitivity, SensitivityResult
from gguf_mixed_quant.type_profiles import (
    BIT_LEVEL_MAP,
    get_bit_level_for_type,
    is_iq_type,
)


# ---------------------------------------------------------------------------
# Multi-level presets matching llama.cpp's quantization presets.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantPreset:
    """A named quantization preset with multiple precision tiers.

    Tiers are ordered from lowest to highest precision.
    Ratios define the weight fraction for each tier (must sum to 1.0).
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


PRESETS: dict[str, QuantPreset] = {
    # --- IQ (importance) quants ---
    "IQ2_XXS": QuantPreset(
        name="IQ2_XXS",
        description="~2-bit importance quants, critical layers bumped to Q4_K/Q5_K",
        tiers=[GGUFQuantType.IQ2_XXS, GGUFQuantType.Q2_K, GGUFQuantType.Q5_K],
        ratios=[0.70, 0.20, 0.10],
    ),
    "IQ2_XS": QuantPreset(
        name="IQ2_XS",
        description="~2.3-bit importance quants, sensitive layers get Q4_K",
        tiers=[GGUFQuantType.IQ2_XS, GGUFQuantType.Q4_K, GGUFQuantType.Q5_K],
        ratios=[0.70, 0.20, 0.10],
    ),
    "IQ2_S": QuantPreset(
        name="IQ2_S",
        description="~2.5-bit importance quants, bumps to IQ3_S/Q4_K",
        tiers=[GGUFQuantType.IQ2_S, GGUFQuantType.IQ3_S, GGUFQuantType.Q4_K],
        ratios=[0.65, 0.25, 0.10],
    ),
    "IQ3_XXS": QuantPreset(
        name="IQ3_XXS",
        description="~3-bit importance quants, sensitive -> Q4_K",
        tiers=[GGUFQuantType.IQ3_XXS, GGUFQuantType.Q4_K, GGUFQuantType.Q5_K],
        ratios=[0.70, 0.20, 0.10],
    ),
    "IQ3_S": QuantPreset(
        name="IQ3_S",
        description="~3.4-bit importance quants, GQA-sensitive -> Q4_K",
        tiers=[GGUFQuantType.IQ3_S, GGUFQuantType.Q4_K, GGUFQuantType.Q5_K],
        ratios=[0.72, 0.18, 0.10],
    ),
    "IQ4_XS": QuantPreset(
        name="IQ4_XS",
        description="~4.25-bit importance quants, critical -> Q5_K",
        tiers=[GGUFQuantType.IQ4_XS, GGUFQuantType.Q5_K, GGUFQuantType.Q6_K],
        ratios=[0.75, 0.15, 0.10],
    ),
    "IQ4_NL": QuantPreset(
        name="IQ4_NL",
        description="~4.5-bit importance quants (non-linear), critical -> Q5_K",
        tiers=[GGUFQuantType.IQ4_NL, GGUFQuantType.Q5_K, GGUFQuantType.Q6_K],
        ratios=[0.75, 0.15, 0.10],
    ),
    # --- K-quants ---
    "Q2_K": QuantPreset(
        name="Q2_K",
        description="2-bit K-quants, attn_v->Q3_K/Q4_K, ffn_down->Q3_K, output->Q6_K",
        tiers=[GGUFQuantType.Q2_K, GGUFQuantType.Q3_K, GGUFQuantType.Q4_K, GGUFQuantType.Q6_K],
        ratios=[0.55, 0.25, 0.12, 0.08],
    ),
    "Q3_K": QuantPreset(
        name="Q3_K",
        description="3-bit K-quants, attn_v->Q4_K/Q5_K, ffn_down->Q4_K/Q5_K",
        tiers=[GGUFQuantType.Q3_K, GGUFQuantType.Q4_K, GGUFQuantType.Q5_K, GGUFQuantType.Q6_K],
        ratios=[0.55, 0.25, 0.13, 0.07],
    ),
    "Q4_K": QuantPreset(
        name="Q4_K",
        description="4-bit K-quants, ~30% sensitive->Q6_K via use_more_bits",
        tiers=[GGUFQuantType.Q4_K, GGUFQuantType.Q5_K, GGUFQuantType.Q6_K],
        ratios=[0.65, 0.20, 0.15],
    ),
    "Q5_K": QuantPreset(
        name="Q5_K",
        description="5-bit K-quants, ~30% sensitive->Q6_K via use_more_bits",
        tiers=[GGUFQuantType.Q5_K, GGUFQuantType.Q6_K],
        ratios=[0.70, 0.30],
    ),
    "Q6_K": QuantPreset(
        name="Q6_K",
        description="6-bit K-quants, output->Q8_0",
        tiers=[GGUFQuantType.Q6_K, GGUFQuantType.Q8_0],
        ratios=[0.92, 0.08],
    ),
    "Q8_0": QuantPreset(
        name="Q8_0",
        description="8-bit, output->F16",
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
        total_bits = sum(assignment.bits_per_weight * assignment.num_weights for assignment in self.assignments)
        total_weights = sum(assignment.num_weights for assignment in self.assignments)
        if total_weights == 0:
            return 0.0
        return total_bits / total_weights

    @property
    def type_distribution(self) -> dict[str, int]:
        """Count of layers per quantization type."""
        dist: dict[str, int] = {}
        for assignment in self.assignments:
            key = assignment.quant_type.value
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


# ---------------------------------------------------------------------------
# Reverse mapping: ggml type name (from llama-quantize output) -> GGUFQuantType
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
    "q3_K": GGUFQuantType.Q3_K,
    "iq4_xs": GGUFQuantType.IQ4_XS,
    "iq4_nl": GGUFQuantType.IQ4_NL,
    "q4_K": GGUFQuantType.Q4_K,
    "q4_0": GGUFQuantType.Q4_K,
    "q4_1": GGUFQuantType.Q4_K,
    "q5_K": GGUFQuantType.Q5_K,
    "q5_0": GGUFQuantType.Q5_K,
    "q5_1": GGUFQuantType.Q5_K,
    "q6_K": GGUFQuantType.Q6_K,
    "q8_0": GGUFQuantType.Q8_0,
    "f16": GGUFQuantType.F16,
}


def _hf_to_gguf_name(hf_name: str) -> str:
    """Convert HuggingFace weight name to GGUF tensor name."""
    import re

    name = re.sub(r"model\.layers\.(\d+)", r"blk.\1", hf_name)
    replacements = {
        # Standard transformer layers
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
        # SSM / Gated DeltaNet (Qwen3.5, Qwen3next)
        "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
        "linear_attn.in_proj_z.weight": "attn_gate.weight",
        "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
        "linear_attn.in_proj_b.weight": "ssm_beta.weight",
        "linear_attn.out_proj.weight": "ssm_out.weight",
        # Mamba-style SSM
        "mamba.in_proj.weight": "ssm_in.weight",
        "mamba.out_proj.weight": "ssm_out.weight",
        # MoE router
        "block_sparse_moe.gate.weight": "ffn_gate_inp.weight",
        "mlp.gate.weight": "ffn_gate_inp.weight",
        "feed_forward.router.weight": "ffn_gate_inp.weight",
    }
    for hf_suffix, gguf_suffix in replacements.items():
        if name.endswith(hf_suffix) or name == hf_suffix:
            prefix = name[: -len(hf_suffix)] if name.endswith(hf_suffix) else ""
            name = prefix + gguf_suffix
            break
    else:
        # MoE experts: model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight
        # -> blk.N.ffn_{gate,up,down}_exps.weight (experts merged in GGUF)
        expert_match = re.match(
            r"(blk\.\d+\.)mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight",
            name,
        )
        if expert_match:
            proj_map = {
                "gate_proj": "ffn_gate_exps",
                "up_proj": "ffn_up_exps",
                "down_proj": "ffn_down_exps",
            }
            name = expert_match.group(1) + proj_map[expert_match.group(2)] + ".weight"
    return name


# Sentinel patterns: tensors that always get the sentinel quant type
# (Q6_K for low-bit presets, Q8_0 for high-bit presets).
# These are excluded from sensitivity banding.
_SENTINEL_PATTERNS: list[str] = [
    "token_embd",     # embedding table
    "output.weight",  # lm_head / output projection
    "ssm_alpha",      # SSM recurrence param
    "ssm_beta",       # SSM recurrence param
    "ffn_gate_exps",  # MoE expert weights
    "ffn_up_exps",    # MoE expert weights
    "ffn_down_exps",  # MoE expert weights
]

# Ignored patterns: tensors that always stay at F16.
# These are excluded from sensitivity banding entirely.
_IGNORED_PATTERNS: list[str] = [
    "ssm_out",        # SSM output projection — critical for hybrid architectures
]


# IQ types that require an importance matrix file
_IQ_NEEDS_IMATRIX: set[GGUFQuantType] = {
    GGUFQuantType.IQ1_S, GGUFQuantType.IQ1_M,
    GGUFQuantType.IQ2_XXS, GGUFQuantType.IQ2_XS, GGUFQuantType.IQ2_S,
    GGUFQuantType.IQ3_XXS,
}

# Nominal bit levels in ascending order
_NOM_LEVELS: list[int] = sorted(BIT_LEVEL_MAP.keys())  # [1, 2, 3, 4, 5, 6, 8]


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: a when t=0, b when t=1."""
    return a + t * (b - a)


def _assign_subtypes_in_band(
    variants: tuple[GGUFQuantType, ...],
    count: int,
    is_k_base: bool,
    is_high_base: bool,
    band_label: str,
    has_imatrix: bool,
    base_nom: int = 4,
    prefer_speed: bool = False,
    variance_ratios: list[float] | None = None,
    iq_cap: int | None = None,
) -> list[GGUFQuantType]:
    """
    Assign IQ/K sub-types within a bit-level band, ordered by ascending sensitivity.

    For bands <= 4b, multiple sub-types exist (e.g. IQ3_XXS, IQ3_S, Q3_K).
    When ``variance_ratios`` is provided, layers with high max/mean variance
    ratio get IQ variants; otherwise default I/K splits apply.  Bands >= 5b
    have a single type and are passed through directly.

    :param variants: Available quant types in this band, ascending BPW.
    :param count: Number of tensors in this band.
    :param is_k_base: True when the baseline preset is K-quant based.
    :param is_high_base: True when base nominal bits >= 5.
    :param band_label: Band identifier ("base-1", "base", "+1", "+2").
    :param has_imatrix: Whether an importance matrix is available.
    :param base_nom: Nominal bit level of the base preset.
    :param prefer_speed: Prefer K-quants over IQ for throughput.
    :param variance_ratios: Per-tensor max/mean variance ratios in
        ascending sensitivity order within this band.
    :param iq_cap: Remaining global IQ budget; None means unlimited.
    :return: Quant types per tensor, ascending sensitivity order.
    """
    if count == 0:
        return []

    # Single-variant bands (>= 5b): all get that type
    if len(variants) == 1:
        return [variants[0]] * count

    # Filter IQ types that need imatrix when unavailable
    available = [v for v in variants if has_imatrix or v not in _IQ_NEEDS_IMATRIX]
    if not available:
        available = [variants[-1]]  # fallback to K-quant

    if len(available) == 1:
        return [available[0]] * count

    iq_vars = [v for v in available if is_iq_type(v)]
    k_vars = [v for v in available if not is_iq_type(v)]

    # No I-quants for high-base (>= Q5_K) or speed preference
    if is_high_base or prefer_speed or not iq_vars:
        k_type = k_vars[-1] if k_vars else available[-1]
        return [k_type] * count

    # --- Determine per-position IQ/K split ---
    if variance_ratios is not None:
        # VR-driven: highest variance ratio → IQ (spikiest activations)
        iq_pct = 0.40
        iq_budget = max(0, round(count * iq_pct))
        iq_budget = min(iq_budget, count - 1)
        if iq_cap is not None:
            iq_budget = min(iq_budget, iq_cap)
        # Indices sorted by descending variance ratio (highest ratio → IQ)
        ranked = sorted(range(count), key=lambda i: variance_ratios[i], reverse=True)
        iq_set = set(ranked[:iq_budget])
        iq_mask = [i in iq_set for i in range(count)]
    elif is_k_base:
        # K-base: generous per-band IQ allocation, capped by the global
        # IQ budget (iq_cap) computed in two_phase_assign.
        iq_pct = 0.40
        iq_budget = max(0, round(count * iq_pct))
        iq_budget = min(iq_budget, count - 1)
        # Apply global IQ cap if provided
        if iq_cap is not None:
            iq_budget = min(iq_budget, iq_cap)
        # IQ sentinels at least-sensitive positions (start of band)
        iq_mask = [i < iq_budget for i in range(count)]
    else:
        # I-base: I-quants fill majority, K at ~10% most-sensitive end
        k_count = max(1, round(count * 0.10))
        iq_count = count - k_count
        iq_mask = [i < iq_count for i in range(count)]

    # --- Assign specific sub-types per position ---
    iq_positions = [i for i in range(count) if iq_mask[i]]
    k_positions = [i for i in range(count) if not iq_mask[i]]

    result: list[GGUFQuantType | None] = [None] * count

    # Distribute variants evenly across positions
    def _spread(positions: list[int], type_list: list[GGUFQuantType],
                sort_key=None) -> None:
        if not positions or not type_list:
            return
        ordered = sorted(positions, key=sort_key) if sort_key else positions
        per_variant = len(ordered) // len(type_list)
        remainder = len(ordered) % len(type_list)
        cursor = 0
        for variant_idx, variant in enumerate(type_list):
            variant_count = per_variant + (1 if variant_idx < remainder else 0)
            for _ in range(variant_count):
                result[ordered[cursor]] = variant
                cursor += 1

    # Spread IQ variants across IQ positions.
    # With VR data: cheapest IQ -> lowest VR, most expensive -> highest VR.
    # Without VR: 85% cheapest variant, remainder spread evenly.
    if iq_positions and iq_vars:
        if len(iq_vars) == 1:
            for idx in iq_positions:
                result[idx] = iq_vars[0]
        elif variance_ratios is not None:
            _spread(iq_positions, iq_vars, sort_key=lambda i: variance_ratios[i])
        else:
            base_count = max(1, round(len(iq_positions) * 0.85))
            for j in range(base_count):
                result[iq_positions[j]] = iq_vars[0]
            if len(iq_vars) > 1:
                _spread(iq_positions[base_count:], iq_vars[1:])

    # Spread K variants across K positions (ascending BPW)
    _spread(k_positions, k_vars)

    # Fill any remaining None (shouldn't happen, but safety)
    fallback = k_vars[-1] if k_vars else available[-1]
    return [t if t is not None else fallback for t in result]


# ---------------------------------------------------------------------------
# Manual tier assignment — user specifies tiers and ratios
# ---------------------------------------------------------------------------


def assign_gguf_types_preset(
    sensitivity_result: SensitivityResult,
    preset_name: str | None = None,
    tiers: list[GGUFQuantType] | None = None,
    ratios: list[float] | None = None,
) -> MixedPrecisionPlan:
    """
    Assign GGUF types using either a named preset or custom tiers + ratios.

    Layers are sorted by sensitivity score. Tier ratios determine what
    fraction of total weights goes into each precision tier. Most-sensitive
    layers get the highest precision tier.

    :param sensitivity_result: Output from compute_sensitivity().
    :param preset_name: Name of a built-in preset (see PRESETS or list_presets()).
    :param tiers: Custom quant types from lowest to highest precision.
    :param ratios: Weight fraction for each tier (must sum to 1.0).
    :return: MixedPrecisionPlan with per-layer assignments.
    """
    if tiers is not None and ratios is not None:
        if len(tiers) != len(ratios):
            raise ValueError(
                f"tiers and ratios must have the same length, "
                f"got {len(tiers)} tiers and {len(ratios)} ratios"
            )
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"ratios must sum to 1.0, got {sum(ratios):.4f}")
        use_tiers = tiers
        use_ratios = ratios
    elif preset_name is not None:
        preset_key = preset_name.upper().replace("-", "_")
        if preset_key not in PRESETS:
            valid = ", ".join(PRESETS.keys())
            raise ValueError(f"Unknown preset: '{preset_name}'. Available: {valid}")
        preset = PRESETS[preset_key]
        use_tiers = preset.tiers
        use_ratios = preset.ratios
    else:
        raise ValueError("Either preset_name or both tiers and ratios must be provided")

    sorted_layers = sensitivity_result.sorted_layers
    total_weights = sum(layer.num_weights for layer in sorted_layers)

    cumulative_thresholds: list[float] = []
    running = 0.0
    for ratio in use_ratios:
        running += ratio
        cumulative_thresholds.append(running)

    assignments: list[LayerAssignment] = []
    accumulated_weights = 0

    for layer in sorted_layers:
        weight_ratio = (accumulated_weights + layer.num_weights) / total_weights
        accumulated_weights += layer.num_weights

        tier_idx = 0
        for i, threshold in enumerate(cumulative_thresholds):
            if weight_ratio <= threshold + 1e-9:
                tier_idx = i
                break
        else:
            tier_idx = len(use_tiers) - 1

        assignments.append(LayerAssignment(
            layer_name=layer.layer_name,
            quant_type=use_tiers[tier_idx],
            score=layer.score,
            num_weights=layer.num_weights,
        ))

    return MixedPrecisionPlan(
        model_id=sensitivity_result.model_id,
        metric=sensitivity_result.metric,
        assignments=assignments,
    )


# ---------------------------------------------------------------------------
# Auto assignment — sensitivity-based band allocation
# ---------------------------------------------------------------------------


def two_phase_assign(
    baseline_map: dict[str, str],
    sensitivity_result: SensitivityResult,
    extra_bpw: float = 0.0,
    has_imatrix: bool = False,
    prefer_speed: bool = False,
    variance_ratios: dict[str, float] | None = None,
    adaptive_bands: bool = False,
) -> MixedPrecisionPlan:
    """
    Sensitivity-based mixed-precision GGUF assignment.

    Ranks weight tensors by sensitivity and assigns them to bit-level
    bands (base, +1, +2, sentinel).  Band sizes are either fixed or
    scaled by the spread of sensitivity scores (``adaptive_bands``).

    Within each band <= 4b, sub-types are assigned using variance ratios
    when available, falling back to default IQ/K percentages.

    Special handling:
      - SSM alpha/beta tensors → sentinel (Q8_0 or Q6_K).
      - SSM output projections → elevated type scaling with preset.
      - MoE expert weights → at least Q8_0.

    :param baseline_map: {gguf_tensor_name: ggml_type} from llama-quantize.
    :param sensitivity_result: Output from compute_sensitivity().
    :param extra_bpw: Shift band boundaries toward higher precision.
    :param has_imatrix: Whether an importance matrix is available.
    :param prefer_speed: Prefer K-quants over IQ for throughput.
    :param variance_ratios: Per-layer max/mean variance ratios keyed by
        HF weight name for IQ/K sub-type assignment.
    :param adaptive_bands: Scale band ratios by sensitivity spread.
    :return: MixedPrecisionPlan with per-layer assignments.
    """
    sorted_layers = sensitivity_result.sorted_layers

    hf_to_gguf = {
        layer.layer_name: _hf_to_gguf_name(layer.layer_name)
        for layer in sorted_layers
    }

    baseline_types: dict[str, GGUFQuantType] = {}
    for gguf_name, ggml_type in baseline_map.items():
        qtype = GGML_TO_QUANT_TYPE.get(ggml_type)
        if qtype is not None:
            baseline_types[gguf_name] = qtype

    matched: list[tuple[LayerSensitivity, GGUFQuantType]] = []
    for layer in sorted_layers:
        gguf_name = hf_to_gguf[layer.layer_name]
        if gguf_name in baseline_types:
            matched.append((layer, baseline_types[gguf_name]))

    if not matched:
        raise ValueError("No scored layers matched baseline tensor names.")

    num_matched = len(matched)

    # Baseline stats
    baseline_bits = sum(
        get_bpw(qtype) * layer.num_weights for layer, qtype in matched
    )
    total_weights = sum(layer.num_weights for layer, _ in matched)

    baseline_dist = Counter(quant_type.value for _, quant_type in matched)
    print(f"  Baseline distribution: {dict(baseline_dist)}")
    print(f"  Baseline avg BPW:     {baseline_bits / total_weights:.3f}")

    # Determine base type from baseline's most common type
    base_type = Counter(quant_type for _, quant_type in matched).most_common(1)[0][0]
    base_nom = get_bit_level_for_type(base_type).nominal_bits
    base_bpw = get_bpw(base_type)
    is_iq_base = is_iq_type(base_type)
    is_high_base = base_nom >= 5

    # Sentinel type: highest-precision override for the most-sensitive
    # bandable tensors.  Must be above the top band.
    # nom ≤ 3: top band is nom+2 ≤ 5, so Q6_K (nom=6) is above.
    # nom ≥ 4: top band reaches nom=6+, so Q8_0 (nom=8) needed.
    if base_nom <= 3:
        sentinel_type = GGUFQuantType.Q6_K
    else:
        sentinel_type = GGUFQuantType.Q8_0

    # Build available bands in ascending order
    base_nom_idx = _NOM_LEVELS.index(base_nom)
    bands: list[tuple[str, tuple[GGUFQuantType, ...]]] = []

    if is_high_base and base_nom_idx > 0:
        bit_level = BIT_LEVEL_MAP[_NOM_LEVELS[base_nom_idx - 1]]
        bands.append(("base-1", bit_level.variants))
    bands.append(("base", BIT_LEVEL_MAP[base_nom].variants))
    if base_nom_idx + 1 < len(_NOM_LEVELS):
        bit_level = BIT_LEVEL_MAP[_NOM_LEVELS[base_nom_idx + 1]]
        bands.append(("+1", bit_level.variants))
    if base_nom_idx + 2 < len(_NOM_LEVELS):
        bit_level = BIT_LEVEL_MAP[_NOM_LEVELS[base_nom_idx + 2]]
        bands.append(("+2", bit_level.variants))

    band_variant_map = {label: variants for label, variants in bands}

    # --- Pre-identify sentinel and ignored tensors ---
    # Sentinel: get the sentinel quant type (excluded from banding)
    # Ignored: stay at F16 (excluded from banding)
    sentinel_indices: set[int] = set()
    ignored_indices: set[int] = set()
    for i, (layer, _) in enumerate(matched):
        gguf_name = hf_to_gguf[layer.layer_name]
        if any(pat in gguf_name for pat in _SENTINEL_PATTERNS):
            sentinel_indices.add(i)
        elif any(pat in gguf_name for pat in _IGNORED_PATTERNS):
            ignored_indices.add(i)

    # Calculate band sizes as fraction of bandable tensors.
    # extra_bpw shifts weight from base toward higher bands.
    bpw_shift = extra_bpw * 0.30  # each +1 bpw shifts ~30% of base up
    n_bandable = num_matched - len(sentinel_indices) - len(ignored_indices)
    sentinel_count = max(1, round(n_bandable * 0.01))
    remaining = n_bandable - sentinel_count

    # Compute spread factor for adaptive banding.
    # spread ∈ [0, 1]: std(normalized scores) / std(Uniform[0,1]).
    # High spread → aggressive (smaller base band), low → conservative.
    if adaptive_bands:
        scores = [layer.score for layer, _ in matched]
        score_min, score_max = min(scores), max(scores)
        score_range = score_max - score_min
        if score_range > 0 and len(scores) > 1:
            normalized = [(s - score_min) / score_range for s in scores]
            normalized_mean = sum(normalized) / len(normalized)
            normalized_variance = sum((x - normalized_mean) ** 2 for x in normalized) / len(normalized)
            std_uniform = 1.0 / 12.0 ** 0.5  # std of Uniform[0,1] ≈ 0.2887
            spread = min(1.0, normalized_variance ** 0.5 / std_uniform)
        else:
            spread = 0.0  # degenerate → all same score → max conservative
        print(f"  Adaptive bands:  spread={spread:.3f}")
    else:
        spread = 1.0  # behaves like original fixed ratios

    if is_high_base:
        ratio_minus1 = 0.08
        ratio_base = max(0.20, _lerp(0.60, 0.47, spread) - max(0.0, bpw_shift))
        ratio_plus1 = min(0.55, _lerp(0.20, 0.30, spread) + max(0.0, bpw_shift))
    else:
        ratio_minus1 = 0.0
        if is_iq_base:
            ratio_base = max(0.25, _lerp(0.80, 0.70, spread) - max(0.0, bpw_shift))
            ratio_plus1 = min(0.55, _lerp(0.12, 0.18, spread) + max(0.0, bpw_shift))
        else:
            ratio_base = max(0.25, _lerp(0.70, 0.45, spread) - max(0.0, bpw_shift))
            ratio_plus1 = min(0.55, _lerp(0.20, 0.35, spread) + max(0.0, bpw_shift))

    band_counts: list[tuple[str, int]] = []
    assigned = 0

    if ratio_minus1 > 0 and "base-1" in band_variant_map:
        minus1_count = round(remaining * ratio_minus1)
        band_counts.append(("base-1", minus1_count))
        assigned += minus1_count

    base_count = round(remaining * ratio_base)
    band_counts.append(("base", base_count))
    assigned += base_count

    if "+1" in band_variant_map:
        plus1_count = round(remaining * ratio_plus1)
        band_counts.append(("+1", plus1_count))
        assigned += plus1_count

    # +2 gets whatever is left (before sentinel)
    plus2_count = remaining - assigned
    if plus2_count > 0 and "+2" in band_variant_map:
        band_counts.append(("+2", plus2_count))
    elif plus2_count > 0:
        # No +2 band available — merge into last existing band
        last_label, last_count = band_counts[-1]
        band_counts[-1] = (last_label, last_count + plus2_count)

    # Print band plan
    print(f"  Base type:  {base_type.value} ({base_bpw:.2f} bpw, nom={base_nom})")
    print(f"  I-base:     {is_iq_base}   High-base: {is_high_base}")
    if sentinel_indices:
        print(f"  Sentinel:   {len(sentinel_indices):3d} tensors (pre-assigned -> {sentinel_type.value})")
    if ignored_indices:
        print(f"  Ignored:    {len(ignored_indices):3d} tensors (pre-assigned -> F16)")
    for label, count in band_counts:
        variants = band_variant_map.get(label, ())
        var_str = " / ".join(v.value for v in variants)
        pct = count / n_bandable * 100 if n_bandable else 0
        print(f"  Band {label:>6}: {count:3d} tensors ({pct:4.0f}%) -> [{var_str}]")
    print(f"  Band  top:  {sentinel_count:3d} tensors ({sentinel_count / n_bandable * 100 if n_bandable else 0:4.0f}%) -> {sentinel_type.value}")

    # Sort bandable tensors by sensitivity ascending
    sensitivity_order = sorted(
        (i for i in range(num_matched) if i not in sentinel_indices and i not in ignored_indices),
        key=lambda i: matched[i][0].score,
    )

    # Assign sub-types per band
    type_assignments: list[GGUFQuantType] = [sentinel_type] * num_matched

    # Pre-assign ignored tensors to F16
    for i in ignored_indices:
        type_assignments[i] = GGUFQuantType.F16

    if is_iq_base:
        iq_remaining = None  # unlimited for IQ-base presets
    elif is_high_base:
        iq_remaining = 0  # no IQ for high-base presets
    else:
        # Quadratic IQ budget: IQ efficiency gains diminish at higher bit
        # levels (IQ2 saves ~21% vs Q2_K, IQ3 ~11%, IQ4 ~6%).
        # nom=2 → ~34%, nom=3 → ~28%, nom=4 → ~6%, nom>=5 → 0%
        iq_total_pct = max(0.0, -0.08 * base_nom**2 + 0.34 * base_nom - 0.02)
        iq_remaining = round(n_bandable * iq_total_pct)
    cursor = 0

    for band_label, count in band_counts:
        variants = band_variant_map.get(band_label)
        if variants is None:
            cursor += count
            continue

        band_indices = sensitivity_order[cursor:cursor + count]

        # Extract per-tensor variance ratios for this band if available
        band_var_ratios: list[float] | None = None
        if variance_ratios is not None:
            band_var_ratios = [
                variance_ratios.get(matched[i][0].layer_name, 1.0)
                for i in band_indices
            ]

        subtypes = _assign_subtypes_in_band(
            variants=variants,
            count=len(band_indices),
            is_k_base=not is_iq_base,
            is_high_base=is_high_base,
            band_label=band_label,
            has_imatrix=has_imatrix,
            base_nom=base_nom,
            prefer_speed=prefer_speed,
            variance_ratios=band_var_ratios,
            iq_cap=iq_remaining,
        )

        # Track IQ usage against cap
        if iq_remaining is not None:
            iq_used = sum(1 for t in subtypes if is_iq_type(t))
            iq_remaining = max(0, iq_remaining - iq_used)

        for idx, subtype in zip(band_indices, subtypes):
            type_assignments[idx] = subtype

        cursor += count

    # Remaining non-override tensors (after all bands) = sentinel
    for i in sensitivity_order[cursor:]:
        type_assignments[i] = sentinel_type

    # Build refined map
    refined: dict[str, GGUFQuantType] = {}
    for i, (layer, _) in enumerate(matched):
        refined[hf_to_gguf[layer.layer_name]] = type_assignments[i]

    # Print results
    refined_dist = Counter(quant_type.value for quant_type in refined.values())
    refined_bits = sum(
        get_bpw(refined[hf_to_gguf[layer.layer_name]]) * layer.num_weights
        for layer, _ in matched
    )
    print(f"  Final distribution: {dict(sorted(refined_dist.items()))}")
    print(f"  Final avg BPW:      {refined_bits / total_weights:.3f}")

    # Build full assignment list (unmatched layers get most common type)
    fallback_type = Counter(quant_type for quant_type in refined.values()).most_common(1)[0][0]
    assignments: list[LayerAssignment] = []
    for layer in sorted_layers:
        gguf_name = hf_to_gguf[layer.layer_name]
        quant_type = refined.get(gguf_name, fallback_type)
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
