"""Map sensitivity scores to GGUF quantization types."""

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
        m = re.match(
            r"(blk\.\d+\.)mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight",
            name,
        )
        if m:
            proj_map = {
                "gate_proj": "ffn_gate_exps",
                "up_proj": "ffn_up_exps",
                "down_proj": "ffn_down_exps",
            }
            name = m.group(1) + proj_map[m.group(2)] + ".weight"
    return name


# Tensor name patterns that should get high-precision (sentinel-level) treatment.
# Router weights are tiny but critical for MoE routing decisions.
# SSM state parameters (alpha/beta) control recurrence dynamics.
_HIGH_PRECISION_PATTERNS: list[str] = [
    "ffn_gate_inp",   # MoE router
    "ssm_alpha",      # SSM recurrence param
    "ssm_beta",       # SSM recurrence param
]

# Tensor name patterns for MoE expert weights
_MOE_EXPERT_PATTERNS: list[str] = [
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
]


# IQ types that require an importance matrix file
_IQ_NEEDS_IMATRIX: set[GGUFQuantType] = {
    GGUFQuantType.IQ1_S, GGUFQuantType.IQ1_M,
    GGUFQuantType.IQ2_XXS, GGUFQuantType.IQ2_XS, GGUFQuantType.IQ2_S,
    GGUFQuantType.IQ3_XXS,
}

# Nominal bit levels in ascending order
_NOM_LEVELS: list[int] = sorted(BIT_LEVEL_MAP.keys())  # [1, 2, 3, 4, 5, 6, 8]


def _assign_subtypes_in_band(
    variants: tuple[GGUFQuantType, ...],
    count: int,
    is_k_base: bool,
    is_high_base: bool,
    band_label: str,
    has_imatrix: bool,
    prefer_speed: bool = False,
    variance_ratios: list[float] | None = None,
    iq_var_threshold: float = 2.0,
    iq_cap: int | None = None,
) -> list[GGUFQuantType]:
    """
    Assign sub-types within a bit-level band, ordered by ascending sensitivity.

    For bands <= 4b, sub-types are ordered by ascending BPW (IQ first, K last).
    Bands >= 5b have a single type.

    When ``variance_ratios`` is provided (per-tensor maxVR/meanVR, ordered by
    ascending sensitivity within this band), tensors with ratio >
    ``iq_var_threshold`` are assigned I-quant variants and others get K-quants.

    Without variance data, defaults apply:
    - K-base presets: ~12% I-quant sentinels per band (5% at +2 tier).
    - I-base presets (IQ1-IQ3): I-quants fill majority, K-quants at ~10%
      most-sensitive end of each band.
    - High-base (>= Q5_K) or prefer_speed: zero I-quants.

    :param variants: Available quant types in this band, ascending BPW.
    :param count: Number of tensors in this band.
    :param is_k_base: True when the baseline preset is K-quant based.
    :param is_high_base: True when base nominal bits >= 5.
    :param band_label: Band identifier ("base-1", "base", "+1", "+2").
    :param has_imatrix: Whether an importance matrix is available.
    :param prefer_speed: Prefer K-quants over IQ for throughput.
    :param variance_ratios: Per-tensor maxVR/meanVR ratios (ascending
        sensitivity order). When provided, drives per-tensor I/K decision.
    :param iq_var_threshold: Ratio above which a tensor gets I-quant.
    :return: List of quant types, one per tensor, ascending sensitivity order.
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

    # --- Determine per-position I/K preference ---
    if variance_ratios is not None:
        # Per-tensor: high variance ratio → I-quant, low → K-quant
        iq_mask = [r > iq_var_threshold for r in variance_ratios]
    elif is_k_base:
        # Default K-base: ~20% IQ sentinels per band, 10% at +2
        iq_pct = 0.10 if band_label == "+2" else 0.20
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

    # Spread IQ variants across IQ positions (ascending BPW)
    if iq_positions and iq_vars:
        per_var = len(iq_positions) // len(iq_vars)
        remainder = len(iq_positions) % len(iq_vars)
        cursor = 0
        for vi, v in enumerate(iq_vars):
            c = per_var + (1 if vi < remainder else 0)
            for j in range(c):
                result[iq_positions[cursor]] = v
                cursor += 1

    # Spread K variants across K positions (ascending BPW)
    if k_positions and k_vars:
        per_var = len(k_positions) // len(k_vars)
        remainder = len(k_positions) % len(k_vars)
        cursor = 0
        for vi, v in enumerate(k_vars):
            c = per_var + (1 if vi < remainder else 0)
            for j in range(c):
                result[k_positions[cursor]] = v
                cursor += 1

    # Fill any remaining None (shouldn't happen, but safety)
    fallback = k_vars[-1] if k_vars else available[-1]
    return [t if t is not None else fallback for t in result]


# ---------------------------------------------------------------------------
# Path 2: Manual tier assignment — user specifies tiers and ratios
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
    for r in use_ratios:
        running += r
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
# Path 1: Auto assignment — sensitivity-based band allocation
# ---------------------------------------------------------------------------


def two_phase_assign(
    baseline_map: dict[str, str],
    sensitivity_result: SensitivityResult,
    extra_bpw: float = 0.0,
    has_imatrix: bool = False,
    prefer_speed: bool = False,
    is_moe: bool = False,
    variance_ratios: dict[str, float] | None = None,
) -> MixedPrecisionPlan:
    """
    Sensitivity-based mixed-precision GGUF assignment.

    Ranks all weight tensors by sensitivity (ascending) and assigns them
    to bit-level bands:

    - Bottom ~55% -> base band (the preset's nominal bit level)
    - Next ~30%   -> +1 band (one bit-level above)
    - Next ~15%   -> +2 band (two bit-levels above)
    - Top ~1%     -> sentinel (Q6_K for base<=4b, Q8_0 for base>=5b)
    - For base>=5b, the very bottom ~8% is demoted to a -1 band.

    Within each band <=4b, sub-types are ordered by ascending BPW
    (e.g. 3b: IQ3_XXS -> IQ3_S -> Q3_K). The I/K split within a band
    is driven per-tensor by ``variance_ratios`` (maxVR/meanVR) when
    available, otherwise by default percentages (5% IQ for K-base,
    majority IQ for I-base).

    :param baseline_map: {gguf_tensor_name: ggml_type} from llama-quantize.
    :param sensitivity_result: Output from compute_sensitivity().
    :param extra_bpw: Shifts band boundaries (positive = more weight in
        higher bands, negative = more in base).
    :param has_imatrix: Whether an importance matrix is available (needed
        for IQ1/IQ2/IQ3_XXS types).
    :param prefer_speed: Prefer K-quants over IQ types for throughput.
    :param is_moe: Model is Mixture-of-Experts (experts always Q8_0).
    :param variance_ratios: Per-layer maxVR/meanVR ratios keyed by HF
        weight name.  When provided, layers with high ratio get I-quant
        variants; otherwise default I/K percentages are used.
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

    n = len(matched)

    # Baseline stats
    baseline_bits = sum(
        get_bpw(qtype) * layer.num_weights for layer, qtype in matched
    )
    total_weights = sum(layer.num_weights for layer, _ in matched)

    baseline_dist = Counter(qt.value for _, qt in matched)
    print(f"  Baseline distribution: {dict(baseline_dist)}")
    print(f"  Baseline avg BPW:     {baseline_bits / total_weights:.3f}")

    # Determine base type from baseline's most common type
    base_type = Counter(qt for _, qt in matched).most_common(1)[0][0]
    base_nom = get_bit_level_for_type(base_type).nominal_bits
    base_bpw = get_bpw(base_type)
    is_iq_base = is_iq_type(base_type)
    is_high_base = base_nom >= 5

    # Sentinel type: Q6_K for base<=4b, Q8_0 for 5-6b, F16 for 8b
    if base_nom <= 4:
        sentinel_type = GGUFQuantType.Q6_K
    elif base_nom <= 6:
        sentinel_type = GGUFQuantType.Q8_0
    else:
        sentinel_type = GGUFQuantType.F16

    # MoE override: sentinel is at least Q8_0
    if is_moe and base_nom <= 6:
        sentinel_type = GGUFQuantType.Q8_0

    # Build available bands in ascending order
    base_nom_idx = _NOM_LEVELS.index(base_nom)
    bands: list[tuple[str, tuple[GGUFQuantType, ...]]] = []

    if is_high_base and base_nom_idx > 0:
        bl = BIT_LEVEL_MAP[_NOM_LEVELS[base_nom_idx - 1]]
        bands.append(("base-1", bl.variants))
    bands.append(("base", BIT_LEVEL_MAP[base_nom].variants))
    if base_nom_idx + 1 < len(_NOM_LEVELS):
        bl = BIT_LEVEL_MAP[_NOM_LEVELS[base_nom_idx + 1]]
        bands.append(("+1", bl.variants))
    if base_nom_idx + 2 < len(_NOM_LEVELS):
        bl = BIT_LEVEL_MAP[_NOM_LEVELS[base_nom_idx + 2]]
        bands.append(("+2", bl.variants))

    band_variant_map = {label: variants for label, variants in bands}

    # --- Pre-identify override tensors (high-prec + MoE experts) ---
    # These are excluded from band assignment so they don't consume IQ budget.
    override_indices: set[int] = set()
    for i, (layer, _) in enumerate(matched):
        gguf_name = hf_to_gguf[layer.layer_name]
        if any(pat in gguf_name for pat in _HIGH_PRECISION_PATTERNS):
            override_indices.add(i)
        if is_moe and any(pat in gguf_name for pat in _MOE_EXPERT_PATTERNS):
            override_indices.add(i)

    # Calculate band sizes as fraction of non-override matched tensors.
    # extra_bpw shifts weight from base toward higher bands.
    bpw_shift = extra_bpw * 0.30  # each +1 bpw shifts ~30% of base up
    n_bandable = n - len(override_indices)
    sentinel_count = max(1, round(n_bandable * 0.01))
    remaining = n_bandable - sentinel_count

    if is_high_base:
        pct_minus1 = 0.08
        pct_base = max(0.20, 0.47 - max(0.0, bpw_shift))
        pct_plus1 = min(0.55, 0.30 + max(0.0, bpw_shift))
    else:
        pct_minus1 = 0.0
        pct_base = max(0.25, 0.55 - max(0.0, bpw_shift))
        pct_plus1 = min(0.55, 0.30 + max(0.0, bpw_shift))

    band_counts: list[tuple[str, int]] = []
    assigned = 0

    if pct_minus1 > 0 and "base-1" in band_variant_map:
        c = round(remaining * pct_minus1)
        band_counts.append(("base-1", c))
        assigned += c

    base_c = round(remaining * pct_base)
    band_counts.append(("base", base_c))
    assigned += base_c

    if "+1" in band_variant_map:
        plus1_c = round(remaining * pct_plus1)
        band_counts.append(("+1", plus1_c))
        assigned += plus1_c

    # +2 gets whatever is left (before sentinel)
    plus2_c = remaining - assigned
    if plus2_c > 0 and "+2" in band_variant_map:
        band_counts.append(("+2", plus2_c))
    elif plus2_c > 0:
        # No +2 band available — merge into last existing band
        last_label, last_count = band_counts[-1]
        band_counts[-1] = (last_label, last_count + plus2_c)

    # Print band plan
    print(f"  Base type:  {base_type.value} ({base_bpw:.2f} bpw, nom={base_nom})")
    print(f"  I-base:     {is_iq_base}   High-base: {is_high_base}")
    if override_indices:
        print(f"  Overrides:  {len(override_indices):3d} tensors (pre-assigned -> {sentinel_type.value})")
    for label, count in band_counts:
        variants = band_variant_map.get(label, ())
        var_str = " / ".join(v.value for v in variants)
        pct = count / n_bandable * 100 if n_bandable else 0
        print(f"  Band {label:>6}: {count:3d} tensors ({pct:4.0f}%) -> [{var_str}]")
    print(f"  Sentinel:   {sentinel_count:3d} tensors ({sentinel_count / n_bandable * 100 if n_bandable else 0:4.0f}%) -> {sentinel_type.value}")

    # Sort non-override tensors by sensitivity ascending (least sensitive first)
    order = sorted(
        (i for i in range(n) if i not in override_indices),
        key=lambda i: matched[i][0].score,
    )

    # Assign sub-types per band
    type_assignments: list[GGUFQuantType] = [sentinel_type] * n

    # Pre-assign override tensors to sentinel (they are not in `order`)
    # (already sentinel from init, but be explicit)

    _MAX_IQ_LAYERS = 25  # Global cap on IQ-assigned layers for K-base
    iq_remaining = _MAX_IQ_LAYERS if not is_iq_base else None
    cursor = 0

    for band_label, count in band_counts:
        variants = band_variant_map.get(band_label)
        if variants is None:
            cursor += count
            continue

        band_indices = order[cursor:cursor + count]

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
    for i in order[cursor:]:
        type_assignments[i] = sentinel_type

    # MoE expert FFN weights → at least Q8_0 when is_moe
    # (experts are already in override_indices and assigned sentinel,
    #  but if sentinel < Q8_0 we bump them)
    if is_moe:
        min_expert_type = GGUFQuantType.Q8_0
        for i in override_indices:
            gguf_name = hf_to_gguf[matched[i][0].layer_name]
            if any(pat in gguf_name for pat in _MOE_EXPERT_PATTERNS):
                if get_bpw(type_assignments[i]) < get_bpw(min_expert_type):
                    type_assignments[i] = min_expert_type

    # Build refined map
    refined: dict[str, GGUFQuantType] = {}
    for i, (layer, _) in enumerate(matched):
        refined[hf_to_gguf[layer.layer_name]] = type_assignments[i]

    # Print results
    refined_dist = Counter(t.value for t in refined.values())
    refined_bits = sum(
        get_bpw(refined[hf_to_gguf[l.layer_name]]) * l.num_weights
        for l, _ in matched
    )
    print(f"  Final distribution: {dict(sorted(refined_dist.items()))}")
    print(f"  Final avg BPW:      {refined_bits / total_weights:.3f}")

    # Build full assignment list (unmatched layers get most common type)
    fallback_type = Counter(qt for qt in refined.values()).most_common(1)[0][0]
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
