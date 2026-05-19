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

_SHARED_EXPERT_PROJ_MAP: dict[str, str] = {
    "gate_proj": "ffn_gate_shexp",
    "up_proj": "ffn_up_shexp",
    "down_proj": "ffn_down_shexp",
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

    # Shared expert: blk.N.mlp.shared_expert.{gate,up,down}_proj.weight
    m = re.match(r"(blk\.\d+\.)mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.weight", name)
    if m:
        return m.group(1) + _SHARED_EXPERT_PROJ_MAP[m.group(2)] + ".weight"

    # Shared expert gate (router): blk.N.mlp.shared_expert_gate.weight
    m = re.match(r"(blk\.\d+\.)mlp\.shared_expert_gate\.weight", name)
    if m:
        return m.group(1) + "ffn_gate_inp_shexp.weight"

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

# Tensors forced to sentinel type (Q6_K/Q8_0). Excluded from banding.
# NOTE: token_embd/output.weight are NOT here — llama.cpp handles them,
# and "output.weight" would substring-match attn_output.weight.
_SENTINEL_PATTERNS: list[str] = [
    "ssm_alpha",     # SSM gating — tiny tensors, critical for state updates
    "ssm_beta",      # SSM gating — tiny tensors, critical for state updates
    "ffn_gate_inp",  # MoE router — tiny but critical for routing decisions
    "_shexp",        # MoE shared experts — small but always-active, keep high precision
    "altup_correct",   # AltUp correction coefficients — tiny, critical for mixing
    "altup_predict",   # AltUp prediction coefficients — tiny, critical for mixing
    "altup_router",    # AltUp router — tiny, critical for routing
]

# Additional patterns forced to sentinel ONLY in MoE models (linear attention + state accumulation)
_MOE_SENTINEL_PATTERNS: list[str] = [
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "attn_qkv",
    "attn_gate",
    "ssm_out",
]

# Tensors forced to minimum base+1 band (residual-stream critical path).
# These are NOT excluded from banding — they participate but cannot go below +1.
_PROMOTED_PATTERNS: list[str] = [
    "attn_output",   # Output projection — direct residual stream contribution
    "ffn_down",      # MLP down projection — direct residual stream contribution
    "attn_v",        # Value projection — feeds into output, critical for content
]

# Tensors forced to F16. Excluded from banding.
_IGNORED_PATTERNS: list[str] = [
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


# ---------------------------------------------------------------------------
# Natural Breaks (Fisher-Jenks) + GVF
# ---------------------------------------------------------------------------


def _fisher_jenks_breaks(sorted_values: list[float], k: int) -> list[int]:
    """Find optimal break indices for *sorted* 1-D data into *k* groups.

    Returns a list of (k-1) break indices: group j contains
    sorted_values[breaks[j-1]:breaks[j]] (with breaks[-1]=0, breaks[k]=n).

    Uses the classic O(n²·k) DP with running sums to avoid recomputing
    class means from scratch.
    """
    n = len(sorted_values)
    if k >= n:
        return list(range(1, n))

    # Precompute prefix sums for O(1) range mean/variance queries
    prefix_sum = [0.0] * (n + 1)
    prefix_sq = [0.0] * (n + 1)
    for i, v in enumerate(sorted_values):
        prefix_sum[i + 1] = prefix_sum[i] + v
        prefix_sq[i + 1] = prefix_sq[i] + v * v

    def _sdcm(lo: int, hi: int) -> float:
        """Sum of squared deviations from class mean for [lo, hi)."""
        cnt = hi - lo
        if cnt <= 0:
            return 0.0
        s = prefix_sum[hi] - prefix_sum[lo]
        sq = prefix_sq[hi] - prefix_sq[lo]
        return sq - s * s / cnt

    INF = float("inf")

    # dp[j][i] = min SDCM partitioning sorted_values[0:i] into j groups
    # backtrack[j][i] = start index of the j-th group in the optimal split
    dp_prev = [INF] * (n + 1)
    back_prev = [0] * (n + 1)

    # Base case: 1 group
    for i in range(1, n + 1):
        dp_prev[i] = _sdcm(0, i)

    backtrack = [[0] * (n + 1) for _ in range(k + 1)]

    for j in range(2, k + 1):
        dp_cur = [INF] * (n + 1)
        back_cur = [0] * (n + 1)
        for i in range(j, n + 1):
            for m in range(j - 1, i):
                cost = dp_prev[m] + _sdcm(m, i)
                if cost < dp_cur[i]:
                    dp_cur[i] = cost
                    back_cur[i] = m
        dp_prev = dp_cur
        back_prev = back_cur
        backtrack[j] = back_cur

    # Reconstruct break indices
    breaks: list[int] = []
    idx = n
    for j in range(k, 1, -1):
        brk = backtrack[j][idx]
        breaks.append(brk)
        idx = brk
    breaks.reverse()
    return breaks


def _total_sdcm(sorted_values: list[float]) -> float:
    """Total sum of squared deviations from the global mean."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    mean = sum(sorted_values) / n
    return sum((v - mean) ** 2 for v in sorted_values)


def _within_sdcm(sorted_values: list[float], breaks: list[int]) -> float:
    """Sum of within-group SDCM given break indices."""
    boundaries = [0] + breaks + [len(sorted_values)]
    total = 0.0
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        group = sorted_values[lo:hi]
        if len(group) <= 1:
            continue
        mean = sum(group) / len(group)
        total += sum((v - mean) ** 2 for v in group)
    return total


def _gvf(sorted_values: list[float], breaks: list[int]) -> float:
    """Goodness of Variance Fit: 1 - within_SDCM / total_SDCM."""
    total = _total_sdcm(sorted_values)
    if total <= 0:
        return 1.0
    return 1.0 - _within_sdcm(sorted_values, breaks) / total


def natural_breaks(
    sorted_scores: list[float],
    num_bands: int,
    gvf_threshold: float = 0.95,
    uplift: float = 0.0,
) -> list[int]:
    """Assign each element of *sorted_scores* to a band index using Natural Breaks.

    1. Try k = 2,3,..num_bands; pick the smallest k where GVF >= gvf_threshold.
    2. If no k reaches the threshold, use k = num_bands.
    3. Map k clusters to band indices 0 .. num_bands-1.
       When k < num_bands, spread clusters to the extremes:
       band = round(cluster * (num_bands - 1) / (k - 1)).

    :param sorted_scores: Ascending-sorted sensitivity scores (after sentinel
        removal and top-1% promotion).
    :param num_bands: Number of available bands (e.g. 3 for base/+1/+2).
    :param gvf_threshold: GVF target (default 0.95).
    :param uplift: Fraction (0.0-1.0) to shift break points leftward, promoting
        more tensors to higher bands. 0.0 = pure NatBreaks, 1.0 = all in top band.
    :return: List of band indices, same length as sorted_scores.
    """
    n = len(sorted_scores)
    if n == 0:
        return []
    if num_bands <= 1 or n == 1:
        return [0] * n

    # Try increasing k, pick smallest that meets GVF threshold
    chosen_k = num_bands
    chosen_breaks: list[int] = []
    for k in range(2, num_bands + 1):
        breaks = _fisher_jenks_breaks(sorted_scores, k)
        gvf_val = _gvf(sorted_scores, breaks)
        if gvf_val >= gvf_threshold:
            chosen_k = k
            chosen_breaks = breaks
            break
    else:
        # No k met threshold; use all bands
        chosen_breaks = _fisher_jenks_breaks(sorted_scores, num_bands)
        chosen_k = num_bands

    # Apply uplift: shift break points leftward to promote tensors to higher bands
    if uplift > 0 and chosen_breaks:
        shifted = [max(1, int(b * (1.0 - uplift))) for b in chosen_breaks]
        # Ensure strictly increasing
        for i in range(1, len(shifted)):
            if shifted[i] <= shifted[i - 1]:
                shifted[i] = shifted[i - 1] + 1
        # Clamp to valid range
        shifted = [min(b, n - 1) for b in shifted]
        chosen_breaks = shifted

    # Assign cluster index per element
    boundaries = [0] + chosen_breaks + [n]
    cluster_per_elem: list[int] = [0] * n
    for c in range(chosen_k):
        for idx in range(boundaries[c], boundaries[c + 1]):
            cluster_per_elem[idx] = c

    # Map clusters -> band indices
    if chosen_k == num_bands:
        return cluster_per_elem

    # Spread: cluster 0 -> band 0, cluster (k-1) -> band (num_bands-1)
    band_per_elem: list[int] = []
    for c in cluster_per_elem:
        if chosen_k == 1:
            band_per_elem.append(0)
        else:
            band_per_elem.append(round(c * (num_bands - 1) / (chosen_k - 1)))
    return band_per_elem


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

    # IQ-base: gradient within band (cheapest IQ → best IQ by sensitivity)
    if is_iq_base:
        if len(iq_vars) == 1:
            return [iq_vars[0]] * count, count
        split = count // 2
        return [iq_vars[0]] * split + [iq_vars[-1]] * (count - split), count

    # K-base: use cheapest IQ for highest-VR positions (non-uniform activations
    # benefit most from imatrix-guided IQ quantization)
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

    # Assign types: single IQ type for IQ positions, K type for the rest
    k_type = k_vars[-1] if k_vars else available[-1]
    iq_type = iq_vars[-1] if iq_vars else available[0]  # best IQ variant (closest to K quality)
    result = [iq_type if i in iq_set else k_type for i in range(count)]

    return result, iq_count


def two_phase_assign(
    baseline_map: dict[str, str],
    sensitivity_result: SensitivityResult,
    extra_bpw: float = 0.0,
    has_imatrix: bool = False,
    prefer_speed: bool = False,
    adaptive_bands: bool = False,
    nb_uplift: float = 0.0,
    k_only: bool = False,
    moe_num_experts: int | None = None,
    moe_num_active: int | None = None,
) -> MixedPrecisionPlan:
    """
    Sensitivity-based mixed-precision assignment.

    Sorts tensors by sensitivity, assigns to bit-level bands (base, +1, +2,
    sentinel). SSM/MoE tensors are pre-assigned to high precision.

    If moe_num_experts and moe_num_active are provided, expert tensor scores
    are scaled by (num_active / num_experts) to reflect sparse activation —
    experts that rarely fire tolerate more compression.

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

    # Scale expert sensitivity scores by MoE activation ratio
    if moe_num_experts and moe_num_active and moe_num_experts > moe_num_active:
        act_ratio = moe_num_active / moe_num_experts
        _expert_re = re.compile(r"\.experts\.\d+\.")
        expert_scaled = 0
        for layer, _ in matched:
            if _expert_re.search(layer.layer_name):
                layer.score *= act_ratio
                expert_scaled += 1
        if expert_scaled:
            print(f"  MoE sparsity: {expert_scaled} expert tensors scaled by {act_ratio:.3f} ({moe_num_active}/{moe_num_experts})")

    num_matched = len(matched)
    total_weights = sum(layer.num_weights for layer, _ in matched)

    # Determine base configuration
    base_type = Counter(qt for _, qt in matched).most_common(1)[0][0]
    base_nom = get_bit_level_for_type(base_type).nominal_bits
    is_iq_base = is_iq_type(base_type)
    is_high_base = base_nom >= 5
    is_moe = bool(moe_num_experts and moe_num_experts > 1)
    # MoE models: sentinel always Q8_0 (shared experts are always-active, critical)
    # Dense models: Q8_0 for base>=5, Q6_K otherwise
    if is_moe or base_nom >= 5:
        sentinel_type = GGUFQuantType.Q8_0
    else:
        sentinel_type = GGUFQuantType.Q6_K

    # Build bands: base, +1, +2, and +3 for ≤3-bit bases
    # For high bases (≥5), also add a -1 band below base for insensitive layers
    base_idx = _NOM_LEVELS.index(base_nom)
    max_offset = 3 if base_nom <= 3 else 2
    # sentinel_cap: bands below this level are allowed; must stay below sentinel
    # MoE uses Q8_0 sentinel so Q6_K can be a regular band
    if base_nom >= 5 or is_moe:
        sentinel_cap = 8  # allows up to Q6_K (nom=6) as regular band
    elif base_nom == 3:
        sentinel_cap = 8
    else:
        sentinel_cap = 6
    band_labels: list[str] = []
    band_variants: dict[str, tuple[GGUFQuantType, ...]] = {}
    def _filter_variants(variants: tuple[GGUFQuantType, ...]) -> tuple[GGUFQuantType, ...]:
        if not k_only:
            return variants
        k = tuple(v for v in variants if not is_iq_type(v))
        return k if k else variants  # fallback to original if no K types

    # Add downward band for high bases (insensitive layers get cheaper quant)
    if base_nom >= 5 and base_idx > 0:
        down_nom = _NOM_LEVELS[base_idx - 1]
        band_labels.append("-1")
        band_variants["-1"] = _filter_variants(BIT_LEVEL_MAP[down_nom].variants)

    band_labels.append("base")
    if is_iq_base:
        # IQ presets: use ONLY the base type (no spread across IQ sub-variants)
        band_variants["base"] = (base_type,)
    else:
        band_variants["base"] = _filter_variants(BIT_LEVEL_MAP[base_nom].variants)
    for offset in range(1, max_offset + 1):
        label = f"+{offset}"
        if base_idx + offset < len(_NOM_LEVELS):
            next_nom = _NOM_LEVELS[base_idx + offset]
            if next_nom >= sentinel_cap:  # don't use sentinel level as a regular band
                break
            band_labels.append(label)
            band_variants[label] = _filter_variants(BIT_LEVEL_MAP[next_nom].variants)

    # Pre-assign sentinel / ignored / promoted tensors
    sentinel_patterns = list(_SENTINEL_PATTERNS)
    if is_moe:
        sentinel_patterns.extend(_MOE_SENTINEL_PATTERNS)
    sentinel_indices: set[int] = set()
    ignored_indices: set[int] = set()
    promoted_indices: set[int] = set()
    for i, (layer, _) in enumerate(matched):
        gguf_name = hf_to_gguf[layer.layer_name]
        if any(pat in gguf_name for pat in sentinel_patterns):
            sentinel_indices.add(i)
        elif any(pat in gguf_name for pat in _IGNORED_PATTERNS):
            ignored_indices.add(i)
        elif any(pat in gguf_name for pat in _PROMOTED_PATTERNS):
            promoted_indices.add(i)

    # Bandable tensors sorted by sensitivity (ascending); promote top 1% to sentinel
    bandable = sorted(
        (i for i in range(num_matched) if i not in sentinel_indices and i not in ignored_indices),
        key=lambda i: matched[i][0].score,
    )
    top_count = max(1, round(len(bandable) * 0.02))
    for i in bandable[-top_count:]:
        sentinel_indices.add(i)
    bandable = bandable[:-top_count]
    n_bandable = len(bandable)

    # Band assignment
    bpw_shift = extra_bpw * 0.30
    bandable_weights = [matched[i][0].num_weights for i in bandable]
    total_bandable_weights = sum(bandable_weights)

    if adaptive_bands:
        # --- Natural Breaks (Fisher-Jenks) + GVF threshold ---
        raw_scores = [matched[i][0].score for i in bandable]
        # Log-transform only when range is extreme (>5000x) to avoid
        # heavy-tail clustering that dumps everything into base band
        score_range = raw_scores[-1] / max(raw_scores[0], 1e-10) if raw_scores else 1
        if score_range > 5000:
            bandable_scores = [math.log(max(s, 1e-10)) for s in raw_scores]
        else:
            bandable_scores = raw_scores
        band_index_per_tensor = natural_breaks(
            bandable_scores, len(band_labels), gvf_threshold=0.98,
            uplift=nb_uplift,
        )

    else:
        # --- Fixed-ratio path ---
        spread = 1.0
        ratios: dict[str, float] = {}
        if is_high_base:
            ratios["base"] = max(0.20, _lerp(0.60, 0.47, spread) - max(0.0, bpw_shift))
            ratios["+1"] = min(0.55, _lerp(0.20, 0.30, spread) + max(0.0, bpw_shift))
        elif len(band_labels) >= 4:
            # 4-band path for ≤2-bit bases: base / +1 / +2 / +3
            ratios["base"] = max(0.25, _lerp(0.48, 0.35, spread) - max(0.0, bpw_shift))
            ratios["+1"] = min(0.40, _lerp(0.25, 0.35, spread) + max(0.0, bpw_shift * 0.5))
            ratios["+2"] = min(0.20, _lerp(0.15, 0.18, spread))
        else:
            ratios["base"] = max(0.25, _lerp(0.58, 0.38, spread) - max(0.0, bpw_shift))
            ratios["+1"] = min(0.55, _lerp(0.25, 0.42, spread) + max(0.0, bpw_shift))

        thresholds: list[float] = []
        running = 0.0
        for label in band_labels[:-1]:
            running += ratios.get(label, 0)
            thresholds.append(running)

        band_index_per_tensor = []
        accumulated = 0
        for w in bandable_weights:
            pct = (accumulated + w) / total_bandable_weights
            accumulated += w
            idx = next((i for i, t in enumerate(thresholds) if pct <= t + 1e-9), len(band_labels) - 1)
            band_index_per_tensor.append(idx)

    # Apply promotion: promoted tensors cannot be in band 0 (base), bump to band 1
    if promoted_indices and len(band_labels) > 1:
        promoted_count = 0
        for pos, orig_idx in enumerate(bandable):
            if orig_idx in promoted_indices and band_index_per_tensor[pos] == 0:
                band_index_per_tensor[pos] = 1
                promoted_count += 1
        if promoted_count:
            print(f"  Promoted: {promoted_count} residual-path tensors bumped to band +1")

    band_counts: list[tuple[str, int]] = [
        (label, band_index_per_tensor.count(i)) for i, label in enumerate(band_labels)
    ]

    # Print plan
    print(f"  Base: {base_type.value} ({get_bpw(base_type):.2f} bpw)")
    if sentinel_indices:
        print(f"  Sentinel: {len(sentinel_indices)} tensors -> {sentinel_type.value}")
    if ignored_indices:
        print(f"  Ignored: {len(ignored_indices)} tensors -> F16")
    cursor_print = 0
    for label, cnt in band_counts:
        band_weight = sum(bandable_weights[cursor_print:cursor_print + cnt])
        pct = band_weight / total_bandable_weights * 100 if total_bandable_weights else 0
        print(f"  Band {label:>6}: {cnt:3d} tensors ({pct:4.1f}% weights) -> [{'/'.join(v.value for v in band_variants[label])}]")
        cursor_print += cnt

    # Assign types per band
    type_assignments: list[GGUFQuantType] = [sentinel_type] * num_matched
    for i in ignored_indices:
        type_assignments[i] = GGUFQuantType.F16

    cursor = 0
    for band_label, count in band_counts:
        indices = bandable[cursor:cursor + count]
        vr_values = [matched[i][0].variance_ratio for i in indices]
        band_vr = (
            [v if v is not None else 1.0 for v in vr_values]
            if any(v is not None for v in vr_values) else None
        )

        iq_cap: int | None = 0 if (is_high_base or prefer_speed) else None
        subtypes, iq_used = _pick_subtypes(
            band_variants[band_label], len(indices), has_imatrix,
            is_iq_base, is_high_base, band_vr, iq_cap,
        )
        for idx, subtype in zip(indices, subtypes):
            type_assignments[idx] = subtype
        cursor += count

    # Build result
    refined = {
        hf_to_gguf[layer.layer_name]: type_assignments[i]
        for i, (layer, _) in enumerate(matched)
    }
    avg_bpw = sum(
        get_bpw(type_assignments[i]) * layer.num_weights
        for i, (layer, _) in enumerate(matched)
    ) / total_weights
    print(f"  Final BPW: {avg_bpw:.3f}")
    print(f"  Distribution: {dict(sorted(Counter(t.value for t in refined.values()).items()))}")

    fallback = Counter(refined.values()).most_common(1)[0][0]
    return MixedPrecisionPlan(
        model_id=sensitivity_result.model_id,
        metric=sensitivity_result.metric,
        assignments=[
            LayerAssignment(
                layer_name=layer.layer_name,
                quant_type=refined.get(hf_to_gguf[layer.layer_name], fallback),
                score=layer.score,
                num_weights=layer.num_weights,
            )
            for layer in sorted_layers
        ],
    )


# Keep old name for CLI compatibility
def list_presets() -> dict[str, str]:
    """No built-in presets in auto mode. Returns empty."""
    return {}


PRESETS: dict = {}
