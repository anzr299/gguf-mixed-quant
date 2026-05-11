"""Assign GGUF quantization types based on sensitivity scores.

Two modes:
  1. Auto (two_phase_assign): ranks tensors by sensitivity, assigns to
     bit-level bands with IQ/K sub-type selection.
  2. Manual (assign_gguf_types_preset): user specifies tiers and ratios.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from gguf_mixed_quant.gguf_types import GGUFQuantType, get_bpw, is_iq_type
from gguf_mixed_quant.sensitivity import LayerSensitivity, SensitivityResult
from gguf_mixed_quant.type_profiles import BIT_LEVEL_MAP, get_bit_level_for_type


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LayerAssignment:
    """Quantization type assigned to a single layer."""

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
        total_bits = sum(a.bits_per_weight * a.num_weights for a in self.assignments)
        total_weights = sum(a.num_weights for a in self.assignments)
        return total_bits / total_weights if total_weights else 0.0

    @property
    def type_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for a in self.assignments:
            key = a.quant_type.value
            dist[key] = dist.get(key, 0) + 1
        return dist

    def summary(self) -> str:
        lines = [
            f"Model: {self.model_id}",
            f"Metric: {self.metric}",
            f"Layers: {len(self.assignments)}",
            f"Avg BPW: {self.avg_bpw:.2f}",
            "Distribution:",
        ]
        for qtype, count in sorted(self.type_distribution.items()):
            lines.append(f"  {qtype}: {count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# HF -> GGUF name mapping
# ---------------------------------------------------------------------------

_HF_TO_GGUF_SUFFIXES: dict[str, str] = {
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
    # SSM / Gated DeltaNet
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
    "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
    "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight",
    # Mamba
    "mamba.in_proj.weight": "ssm_in.weight",
    "mamba.out_proj.weight": "ssm_out.weight",
    # MoE router
    "block_sparse_moe.gate.weight": "ffn_gate_inp.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight",
    "feed_forward.router.weight": "ffn_gate_inp.weight",
}

_MOE_PROJ_MAP: dict[str, str] = {
    "gate_proj": "ffn_gate_exps",
    "up_proj": "ffn_up_exps",
    "down_proj": "ffn_down_exps",
}


def _hf_to_gguf_name(hf_name: str) -> str:
    """Convert HuggingFace weight name to GGUF tensor name."""
    name = re.sub(r"model\.layers\.(\d+)", r"blk.\1", hf_name)

    for hf_suffix, gguf_suffix in _HF_TO_GGUF_SUFFIXES.items():
        if name.endswith(hf_suffix) or name == hf_suffix:
            prefix = name[:-len(hf_suffix)] if name.endswith(hf_suffix) else ""
            return prefix + gguf_suffix

    # MoE experts: blk.N.mlp.experts.E.{gate,up,down}_proj.weight
    m = re.match(r"(blk\.\d+\.)mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight", name)
    if m:
        return m.group(1) + _MOE_PROJ_MAP[m.group(2)] + ".weight"

    return name


# ---------------------------------------------------------------------------
# ggml type string -> GGUFQuantType
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


# ---------------------------------------------------------------------------
# Sentinel / ignored patterns
# ---------------------------------------------------------------------------

# Tensors forced to sentinel type (Q6_K or Q8_0). Excluded from banding.
# NOTE: token_embd/output.weight are NOT here — llama.cpp handles them,
# and "output.weight" would substring-match attn_output.weight.
_SENTINEL_PATTERNS: list[str] = [
    "ssm_alpha",
    "ssm_beta",
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
]

# Tensors forced to F16. Excluded from banding.
_IGNORED_PATTERNS: list[str] = [
    "ssm_out",
]

# IQ types that need an importance matrix file.
_IQ_NEEDS_IMATRIX: frozenset[GGUFQuantType] = frozenset({
    GGUFQuantType.IQ1_S, GGUFQuantType.IQ1_M,
    GGUFQuantType.IQ2_XXS, GGUFQuantType.IQ2_XS, GGUFQuantType.IQ2_S,
    GGUFQuantType.IQ3_XXS,
})


# ---------------------------------------------------------------------------
# Manual assignment
# ---------------------------------------------------------------------------


def assign_gguf_types_preset(
    sensitivity_result: SensitivityResult,
    tiers: list[GGUFQuantType],
    ratios: list[float],
) -> MixedPrecisionPlan:
    """
    Assign types by splitting layers into tiers by weight fraction.

    Layers sorted by sensitivity; lowest-sensitivity layers get the
    cheapest tier, highest get the most expensive.

    :param sensitivity_result: Output from compute_sensitivity().
    :param tiers: Quant types from lowest to highest precision.
    :param ratios: Fraction of total weights per tier (must sum to 1.0).
    :return: MixedPrecisionPlan.
    """
    if len(tiers) != len(ratios):
        raise ValueError(f"tiers ({len(tiers)}) and ratios ({len(ratios)}) must match")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios):.4f}")

    sorted_layers = sensitivity_result.sorted_layers
    total_weights = sum(layer.num_weights for layer in sorted_layers)

    # Cumulative thresholds
    thresholds: list[float] = []
    running = 0.0
    for r in ratios:
        running += r
        thresholds.append(running)

    assignments: list[LayerAssignment] = []
    accumulated = 0
    for layer in sorted_layers:
        weight_pct = (accumulated + layer.num_weights) / total_weights
        accumulated += layer.num_weights
        tier_idx = next((i for i, t in enumerate(thresholds) if weight_pct <= t + 1e-9), len(tiers) - 1)
        assignments.append(LayerAssignment(
            layer_name=layer.layer_name,
            quant_type=tiers[tier_idx],
            score=layer.score,
            num_weights=layer.num_weights,
        ))

    return MixedPrecisionPlan(
        model_id=sensitivity_result.model_id,
        metric=sensitivity_result.metric,
        assignments=assignments,
    )


# ---------------------------------------------------------------------------
# Auto assignment (two-phase)
# ---------------------------------------------------------------------------

_NOM_LEVELS: list[int] = sorted(BIT_LEVEL_MAP.keys())


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def _compute_spread(scores: list[float]) -> float:
    """Normalized spread: std(scores) / std(Uniform[0,1]).

    Filters non-finite values. Caller is responsible for excluding
    sentinel/ignored scores before passing them in.
    """
    finite = sorted(s for s in scores if math.isfinite(s))
    if len(finite) < 2:
        return 0.0
    lo, hi = finite[0], finite[-1]
    rng = hi - lo
    if rng <= 0:
        return 0.0
    norm = [(s - lo) / rng for s in finite]
    mean = sum(norm) / len(norm)
    var = sum((x - mean) ** 2 for x in norm) / len(norm)
    return min(1.0, var ** 0.5 / (1.0 / 12.0 ** 0.5))


def _pick_subtypes(
    variants: tuple[GGUFQuantType, ...],
    count: int,
    has_imatrix: bool,
    is_iq_base: bool,
    is_high_base: bool,
    variance_ratios: list[float] | None,
    iq_cap: int | None,
) -> tuple[list[GGUFQuantType], int]:
    """
    Pick sub-types within a band for `count` tensors (ordered least→most sensitive).

    Returns (type_list, iq_used).
    """
    if count == 0:
        return [], 0
    if len(variants) == 1:
        return [variants[0]] * count, 0

    # Filter unavailable IQ types
    available = [v for v in variants if has_imatrix or v not in _IQ_NEEDS_IMATRIX]
    if not available:
        available = [variants[-1]]
    if len(available) == 1:
        return [available[0]] * count, 0

    iq_vars = [v for v in available if is_iq_type(v)]
    k_vars = [v for v in available if not is_iq_type(v)]

    # High-base (>=5b) or no IQ variants: all K-quant
    if is_high_base or not iq_vars:
        return [k_vars[-1] if k_vars else available[-1]] * count, 0

    # IQ-base: best IQ variant in every band — precision promotion is via banding only
    if is_iq_base:
        return [iq_vars[-1]] * count, count

    # K-base: 40% IQ for cheapest positions
    iq_count = round(count * 0.40)

    if iq_cap is not None:
        iq_count = min(iq_count, iq_cap)
    iq_count = min(iq_count, count - 1)
    iq_count = max(0, iq_count)

    # If variance ratios provided, highest-VR positions get IQ
    if variance_ratios is not None and iq_count > 0:
        ranked = sorted(range(count), key=lambda i: variance_ratios[i], reverse=True)
        iq_set = set(ranked[:iq_count])
    else:
        # Default: IQ at least-sensitive (start of band)
        iq_set = set(range(iq_count))

    # Assign types
    iq_type = iq_vars[0]  # cheapest IQ variant
    k_type = k_vars[-1] if k_vars else available[-1]  # most expensive K variant
    result = [iq_type if i in iq_set else k_type for i in range(count)]

    return result, iq_count


def two_phase_assign(
    baseline_map: dict[str, str],
    sensitivity_result: SensitivityResult,
    extra_bpw: float = 0.0,
    has_imatrix: bool = False,
    prefer_speed: bool = False,
    adaptive_bands: bool = False,
) -> MixedPrecisionPlan:
    """
    Sensitivity-based mixed-precision assignment.

    Sorts tensors by sensitivity, assigns to bit-level bands (base, +1, +2,
    sentinel). SSM/MoE tensors are pre-assigned to high precision.

    :param baseline_map: {gguf_tensor_name: ggml_type} from llama-quantize.
    :param sensitivity_result: Output from compute_sensitivity().
    :param extra_bpw: Shift bands toward higher precision.
    :param has_imatrix: Whether an importance matrix is available.
    :param prefer_speed: Prefer K-quants over IQ.
    :param adaptive_bands: Scale band sizes by sensitivity spread.
    :return: MixedPrecisionPlan.
    """
    sorted_layers = sensitivity_result.sorted_layers
    hf_to_gguf = {l.layer_name: _hf_to_gguf_name(l.layer_name) for l in sorted_layers}

    # Parse baseline types
    baseline_types: dict[str, GGUFQuantType] = {}
    for name, ggml_type in baseline_map.items():
        qtype = GGML_TO_QUANT_TYPE.get(ggml_type)
        if qtype is not None:
            baseline_types[name] = qtype

    # Match scored layers to baseline
    matched: list[tuple[LayerSensitivity, GGUFQuantType]] = []
    for layer in sorted_layers:
        gguf_name = hf_to_gguf[layer.layer_name]
        if gguf_name in baseline_types:
            matched.append((layer, baseline_types[gguf_name]))

    if not matched:
        raise ValueError("No scored layers matched baseline tensor names.")

    num_matched = len(matched)
    total_weights = sum(layer.num_weights for layer, _ in matched)

    # Determine base type from most common in baseline
    base_type = Counter(qt for _, qt in matched).most_common(1)[0][0]
    base_nom = get_bit_level_for_type(base_type).nominal_bits
    base_bpw = get_bpw(base_type)
    is_iq_base = is_iq_type(base_type)
    is_high_base = base_nom >= 5

    # Sentinel type: above top band
    sentinel_type = GGUFQuantType.Q6_K if base_nom <= 3 else GGUFQuantType.Q8_0

    # Build bands
    base_idx = _NOM_LEVELS.index(base_nom)
    bands: list[tuple[str, tuple[GGUFQuantType, ...]]] = []
    if is_high_base and base_idx > 0:
        bands.append(("base-1", BIT_LEVEL_MAP[_NOM_LEVELS[base_idx - 1]].variants))
    bands.append(("base", BIT_LEVEL_MAP[base_nom].variants))
    if base_idx + 1 < len(_NOM_LEVELS):
        bands.append(("+1", BIT_LEVEL_MAP[_NOM_LEVELS[base_idx + 1]].variants))
    if base_idx + 2 < len(_NOM_LEVELS):
        bands.append(("+2", BIT_LEVEL_MAP[_NOM_LEVELS[base_idx + 2]].variants))
    band_variants = dict(bands)

    # Identify sentinel / ignored tensors
    sentinel_indices: set[int] = set()
    ignored_indices: set[int] = set()
    for i, (layer, _) in enumerate(matched):
        gguf_name = hf_to_gguf[layer.layer_name]
        if any(pat in gguf_name for pat in _SENTINEL_PATTERNS):
            sentinel_indices.add(i)
        elif any(pat in gguf_name for pat in _IGNORED_PATTERNS):
            ignored_indices.add(i)

    # Band sizing
    n_bandable = num_matched - len(sentinel_indices) - len(ignored_indices)
    top_sentinel_count = max(1, round(n_bandable * 0.01))

    # Move top 1% most sensitive bandable layers into sentinel
    bandable_by_score = sorted(
        (i for i in range(num_matched) if i not in sentinel_indices and i not in ignored_indices),
        key=lambda i: matched[i][0].score,
    )
    for i in bandable_by_score[-top_sentinel_count:]:
        sentinel_indices.add(i)

    # Recount after promoting top 1%
    n_bandable -= top_sentinel_count
    remaining = n_bandable

    # Adaptive spread — only over bandable scores (sentinel/ignored already assigned)
    bpw_shift = extra_bpw * 0.30
    if adaptive_bands:
        bandable_scores = [
            matched[i][0].score for i in range(num_matched)
            if i not in sentinel_indices and i not in ignored_indices
        ]
        spread = _compute_spread(bandable_scores)
        print(f"  Spread: {spread:.3f}")
    else:
        spread = 1.0

    # Compute band ratios
    if is_high_base:
        r_minus1 = 0.08
        r_base = max(0.20, _lerp(0.60, 0.47, spread) - max(0.0, bpw_shift))
        r_plus1 = min(0.55, _lerp(0.20, 0.30, spread) + max(0.0, bpw_shift))
    else:
        r_minus1 = 0.0
        r_base = max(0.25, _lerp(0.70, 0.45, spread) - max(0.0, bpw_shift))
        r_plus1 = min(0.55, _lerp(0.20, 0.35, spread) + max(0.0, bpw_shift))

    # Allocate counts
    band_counts: list[tuple[str, int]] = []
    assigned = 0
    if r_minus1 > 0 and "base-1" in band_variants:
        c = round(remaining * r_minus1)
        band_counts.append(("base-1", c))
        assigned += c
    c = round(remaining * r_base)
    band_counts.append(("base", c))
    assigned += c
    if "+1" in band_variants:
        c = round(remaining * r_plus1)
        band_counts.append(("+1", c))
        assigned += c
    leftover = remaining - assigned
    if leftover > 0 and "+2" in band_variants:
        band_counts.append(("+2", leftover))
    elif leftover > 0:
        label, cnt = band_counts[-1]
        band_counts[-1] = (label, cnt + leftover)

    # Print plan
    print(f"  Base: {base_type.value} ({base_bpw:.2f} bpw)")
    if sentinel_indices:
        print(f"  Sentinel: {len(sentinel_indices)} tensors -> {sentinel_type.value}")
    if ignored_indices:
        print(f"  Ignored: {len(ignored_indices)} tensors -> F16")
    for label, cnt in band_counts:
        variants = band_variants.get(label, ())
        pct = cnt / n_bandable * 100 if n_bandable else 0
        print(f"  Band {label:>6}: {cnt:3d} ({pct:4.0f}%) -> [{'/'.join(v.value for v in variants)}]")

    # Sort bandable tensors by sensitivity (ascending)
    sensitivity_order = sorted(
        (i for i in range(num_matched) if i not in sentinel_indices and i not in ignored_indices),
        key=lambda i: matched[i][0].score,
    )

    # Assign types
    type_assignments: list[GGUFQuantType] = [sentinel_type] * num_matched
    for i in ignored_indices:
        type_assignments[i] = GGUFQuantType.F16

    # IQ budget for K-base presets
    if is_iq_base:
        iq_remaining: int | None = None
    elif is_high_base or prefer_speed:
        iq_remaining = 0
    else:
        iq_pct = max(0.0, -0.08 * base_nom**2 + 0.34 * base_nom - 0.02)
        iq_remaining = round(n_bandable * iq_pct)

    cursor = 0
    for band_label, count in band_counts:
        variants = band_variants.get(band_label)
        if variants is None:
            cursor += count
            continue

        band_indices = sensitivity_order[cursor:cursor + count]

        # Per-tensor variance ratios for this band (from LayerSensitivity)
        has_vr = any(matched[i][0].variance_ratio is not None for i in band_indices)
        band_vr: list[float] | None = None
        if has_vr:
            band_vr = [matched[i][0].variance_ratio or 1.0 for i in band_indices]

        subtypes, iq_used = _pick_subtypes(
            variants=variants,
            count=len(band_indices),
            has_imatrix=has_imatrix,
            is_iq_base=is_iq_base,
            is_high_base=is_high_base,
            variance_ratios=band_vr,
            iq_cap=iq_remaining,
        )

        if iq_remaining is not None:
            iq_remaining = max(0, iq_remaining - iq_used)

        for idx, subtype in zip(band_indices, subtypes):
            type_assignments[idx] = subtype
        cursor += count

    # Build result
    refined: dict[str, GGUFQuantType] = {}
    for i, (layer, _) in enumerate(matched):
        refined[hf_to_gguf[layer.layer_name]] = type_assignments[i]

    refined_bits = sum(
        get_bpw(refined[hf_to_gguf[layer.layer_name]]) * layer.num_weights
        for layer, _ in matched
    )
    print(f"  Final BPW: {refined_bits / total_weights:.3f}")
    print(f"  Distribution: {dict(sorted(Counter(t.value for t in refined.values()).items()))}")

    # Build assignments for all scored layers
    fallback = Counter(refined.values()).most_common(1)[0][0]
    assignments = [
        LayerAssignment(
            layer_name=layer.layer_name,
            quant_type=refined.get(hf_to_gguf[layer.layer_name], fallback),
            score=layer.score,
            num_weights=layer.num_weights,
        )
        for layer in sorted_layers
    ]

    return MixedPrecisionPlan(
        model_id=sensitivity_result.model_id,
        metric=sensitivity_result.metric,
        assignments=assignments,
    )


# Keep old name for CLI compatibility
def list_presets() -> dict[str, str]:
    """No built-in presets in auto mode. Returns empty."""
    return {}


PRESETS: dict = {}
