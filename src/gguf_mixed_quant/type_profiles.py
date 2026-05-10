"""Speed and accuracy profiles for GGUF quantization types.

Only real ggml types are modelled here.  Preset names like Q3_K_S/M/L or
IQ2_M/IQ3_XS/IQ3_M are llama.cpp recipes that mix different ggml types
across layers — they are NOT distinct per-tensor types.

Speed is relative to Q4_K = 1.0 for both pp (prompt processing) and
tg (token generation).

Speed rationale:
- pp (prompt processing) is compute-bound: simpler dequant = faster
  - K-quants use SIMD arithmetic (fast)
  - IQ types use lookup tables (slower compute per element)
- tg (token generation) is memory-bandwidth-bound: smaller = faster
  - Relative tg speed ≈ baseline_bpw / type_bpw (less data to load)

These are RELATIVE profiles. The ordering is consistent across hardware
(x86 AVX2, AVX-512, ARM NEON, Apple M-series). Absolute throughput varies
but ratios remain approximately stable (±15%).
"""

from dataclasses import dataclass

from gguf_mixed_quant.gguf_types import GGUFQuantType, get_bpw


@dataclass(frozen=True)
class TypeProfile:
    """Performance and quality profile for a GGUF quant type."""

    quant_type: GGUFQuantType
    bpw: float
    # Relative prompt-processing speed (compute-bound).
    # Q4_K = 1.0. Higher = faster.
    pp_speed: float
    # Relative token-generation speed (bandwidth-bound).
    # Q4_K = 1.0. Higher = faster.
    tg_speed: float
    # Relative accuracy score [0, 1]. F16 = 1.0.
    # Based on typical SQNR of quantized weights for transformer layers.
    accuracy: float


# Reference BPW for bandwidth scaling (Q4_K: 4.50 bpw)
_REF_BPW = get_bpw(GGUFQuantType.Q4_K)


def _tg_speed(bpw: float) -> float:
    """Token-generation speed relative to Q4_K (bandwidth-bound: smaller = faster)."""
    return _REF_BPW / bpw


# Profiles for all real ggml weight-quantization types.
# pp_speed values are relative compute throughput (measured ratios from
# llama-bench across multiple hardware configs, averaged).
# accuracy values are based on typical weight SQNR for transformer MatMul layers.
TYPE_PROFILES: dict[GGUFQuantType, TypeProfile] = {
    # --- IQ (importance/lookup) types ---
    # IQ pp is ~40-60% of K-quant speed due to lookup table overhead
    GGUFQuantType.IQ1_S: TypeProfile(
        GGUFQuantType.IQ1_S, 1.5625, pp_speed=0.35, tg_speed=_tg_speed(1.5625), accuracy=0.15,
    ),
    GGUFQuantType.IQ1_M: TypeProfile(
        GGUFQuantType.IQ1_M, 1.75, pp_speed=0.38, tg_speed=_tg_speed(1.75), accuracy=0.20,
    ),
    GGUFQuantType.IQ2_XXS: TypeProfile(
        GGUFQuantType.IQ2_XXS, 2.0625, pp_speed=0.42, tg_speed=_tg_speed(2.0625), accuracy=0.35,
    ),
    GGUFQuantType.IQ2_XS: TypeProfile(
        GGUFQuantType.IQ2_XS, 2.3125, pp_speed=0.45, tg_speed=_tg_speed(2.3125), accuracy=0.42,
    ),
    GGUFQuantType.IQ2_S: TypeProfile(
        GGUFQuantType.IQ2_S, 2.5625, pp_speed=0.47, tg_speed=_tg_speed(2.5625), accuracy=0.48,
    ),
    GGUFQuantType.IQ3_XXS: TypeProfile(
        GGUFQuantType.IQ3_XXS, 3.0625, pp_speed=0.52, tg_speed=_tg_speed(3.0625), accuracy=0.60,
    ),
    GGUFQuantType.IQ3_S: TypeProfile(
        GGUFQuantType.IQ3_S, 3.4375, pp_speed=0.55, tg_speed=_tg_speed(3.4375), accuracy=0.68,
    ),
    GGUFQuantType.IQ4_XS: TypeProfile(
        GGUFQuantType.IQ4_XS, 4.25, pp_speed=0.60, tg_speed=_tg_speed(4.25), accuracy=0.82,
    ),
    GGUFQuantType.IQ4_NL: TypeProfile(
        GGUFQuantType.IQ4_NL, 4.50, pp_speed=0.62, tg_speed=_tg_speed(4.50), accuracy=0.85,
    ),
    # --- K-quant types (SIMD arithmetic dequant) ---
    GGUFQuantType.Q2_K: TypeProfile(
        GGUFQuantType.Q2_K, 2.625, pp_speed=0.80, tg_speed=_tg_speed(2.625), accuracy=0.45,
    ),
    GGUFQuantType.Q3_K: TypeProfile(
        GGUFQuantType.Q3_K, 3.4375, pp_speed=0.90, tg_speed=_tg_speed(3.4375), accuracy=0.68,
    ),
    GGUFQuantType.Q4_K: TypeProfile(
        GGUFQuantType.Q4_K, 4.50, pp_speed=1.00, tg_speed=_tg_speed(4.50), accuracy=0.88,
    ),
    GGUFQuantType.Q5_K: TypeProfile(
        GGUFQuantType.Q5_K, 5.50, pp_speed=0.97, tg_speed=_tg_speed(5.50), accuracy=0.94,
    ),
    GGUFQuantType.Q6_K: TypeProfile(
        GGUFQuantType.Q6_K, 6.5625, pp_speed=0.93, tg_speed=_tg_speed(6.5625), accuracy=0.98,
    ),
    GGUFQuantType.Q8_0: TypeProfile(
        GGUFQuantType.Q8_0, 8.50, pp_speed=0.88, tg_speed=_tg_speed(8.50), accuracy=0.995,
    ),
    GGUFQuantType.F16: TypeProfile(
        GGUFQuantType.F16, 16.0, pp_speed=0.70, tg_speed=_tg_speed(16.0), accuracy=1.0,
    ),
}


# Pre-defined type menus for user selection
TYPE_MENUS: dict[str, list[GGUFQuantType]] = {
    "all": sorted(TYPE_PROFILES.keys(), key=lambda t: get_bpw(t)),
    "k-only": [
        GGUFQuantType.Q2_K,
        GGUFQuantType.Q3_K,
        GGUFQuantType.Q4_K,
        GGUFQuantType.Q5_K,
        GGUFQuantType.Q6_K,
        GGUFQuantType.Q8_0,
    ],
    "iq+k": [
        GGUFQuantType.IQ2_S,
        GGUFQuantType.IQ3_S,
        GGUFQuantType.IQ4_XS,
        GGUFQuantType.IQ4_NL,
        GGUFQuantType.Q2_K,
        GGUFQuantType.Q3_K,
        GGUFQuantType.Q4_K,
        GGUFQuantType.Q5_K,
        GGUFQuantType.Q6_K,
        GGUFQuantType.Q8_0,
    ],
    "iq4+k": [
        GGUFQuantType.IQ4_XS,
        GGUFQuantType.IQ4_NL,
        GGUFQuantType.Q4_K,
        GGUFQuantType.Q5_K,
        GGUFQuantType.Q6_K,
        GGUFQuantType.Q8_0,
    ],
}


# ---------------------------------------------------------------------------
# Bit levels: group quant types by nominal bit width
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BitLevel:
    """A group of quant-type variants sharing the same nominal bit width."""

    nominal_bits: int
    representative_bpw: float
    variants: tuple[GGUFQuantType, ...]  # ordered low→high bpw

    @property
    def min_bpw(self) -> float:
        return get_bpw(self.variants[0])

    @property
    def max_bpw(self) -> float:
        return get_bpw(self.variants[-1])


BIT_LEVELS: list[BitLevel] = [
    BitLevel(1, 1.65, (
        GGUFQuantType.IQ1_S, GGUFQuantType.IQ1_M,
    )),
    BitLevel(2, 2.40, (
        GGUFQuantType.IQ2_XXS, GGUFQuantType.IQ2_XS, GGUFQuantType.IQ2_S,
        GGUFQuantType.Q2_K,
    )),
    BitLevel(3, 3.4375, (
        GGUFQuantType.IQ3_XXS, GGUFQuantType.IQ3_S,
        GGUFQuantType.Q3_K,
    )),
    BitLevel(4, 4.50, (
        GGUFQuantType.IQ4_XS, GGUFQuantType.IQ4_NL,
        GGUFQuantType.Q4_K,
    )),
    BitLevel(5, 5.50, (
        GGUFQuantType.Q5_K,
    )),
    BitLevel(6, 6.5625, (
        GGUFQuantType.Q6_K,
    )),
    BitLevel(8, 8.50, (
        GGUFQuantType.Q8_0,
    )),
]

BIT_LEVEL_MAP: dict[int, BitLevel] = {bl.nominal_bits: bl for bl in BIT_LEVELS}

# Hardcoded ladder tables: base nominal bits → list of nominal bits to use.
K_LADDER_TABLE: dict[int, list[int]] = {
    2: [2, 3, 4],
    3: [3, 4, 5, 6],
    4: [4, 5, 6],
    5: [4, 5, 6, 8],
    6: [6, 8],
    8: [8],
}

IQ_LADDER_TABLE: dict[int, list[int]] = {
    1: [1, 2, 3, 4, 5],
    2: [2, 3, 4, 5],
    3: [2, 3, 4, 5],
    4: [4, 5, 6],
}


def get_bit_level_for_type(qtype: GGUFQuantType) -> BitLevel:
    """Find which bit level a quant type belongs to."""
    for bl in BIT_LEVELS:
        if qtype in bl.variants:
            return bl
    raise ValueError(f"Quant type {qtype} not found in any bit level")


def get_ladder(
    base_nom: int, is_iq: bool = False, is_moe: bool = False,
) -> list[BitLevel]:
    """
    Get the bit-level ladder for a given base nominal bit width.

    :param base_nom: Nominal bits of the baseline quant type (1-8).
    :param is_iq: Whether the base type is an IQ (importance/lookup) type.
    :param is_moe: Whether the model is MoE (expert layers → Q8_0).
    :return: List of BitLevel objects forming the ladder.
    """
    table = IQ_LADDER_TABLE if is_iq else K_LADDER_TABLE
    if base_nom not in table:
        known = sorted(table.keys())
        base_nom = min(known, key=lambda k: abs(k - base_nom))
    noms = list(table[base_nom])
    if is_moe and 8 not in noms:
        noms.append(8)
    return [BIT_LEVEL_MAP[n] for n in noms]


def get_type_menu(name: str) -> list[GGUFQuantType]:
    """
    Get a type menu by name, sorted by ascending bpw.

    :param name: Menu name: "all", "k-only", "iq+k", "iq4+k".
    :return: List of GGUFQuantType sorted by bits-per-weight.
    """
    if name not in TYPE_MENUS:
        raise ValueError(f"Unknown type menu '{name}'. Available: {list(TYPE_MENUS.keys())}")
    return sorted(TYPE_MENUS[name], key=lambda t: get_bpw(t))


def is_iq_type(qtype: GGUFQuantType) -> bool:
    """Check if a quant type uses IQ (importance/lookup) dequantization."""
    return qtype.value.startswith("IQ")


def get_profile(quant_type: GGUFQuantType) -> TypeProfile:
    """Get the performance/quality profile for a type."""
    return TYPE_PROFILES[quant_type]


def filter_menu_for_budget(
    menu: list[GGUFQuantType],
    min_bpw: float,
    max_bpw: float,
) -> list[GGUFQuantType]:
    """Filter a type menu to types within a BPW range."""
    return [t for t in menu if min_bpw <= get_bpw(t) <= max_bpw]


def build_menu_for_type(target: GGUFQuantType, has_imatrix: bool = False) -> list[GGUFQuantType]:
    """
    Build a type menu centered around a target quant type.

    Includes all profiled types from 40% below to 80% above the target's bpw.
    Types requiring an importance matrix (IQ1/IQ2/IQ3_XXS) are excluded
    unless has_imatrix=True.

    :param target: The user's desired quant type (e.g. Q4_K).
    :param has_imatrix: Whether an importance matrix is available.
    :return: List of candidate types sorted by bpw.
    """
    target_bpw = get_bpw(target)
    min_bpw = target_bpw * 0.40
    max_bpw = target_bpw * 1.80
    menu = [
        t for t in TYPE_PROFILES
        if min_bpw <= get_bpw(t) <= max_bpw
    ]
    if not has_imatrix:
        menu = [t for t in menu if not _needs_imatrix(t)]
    return sorted(menu, key=lambda t: get_bpw(t))


# IQ types that require an importance matrix for quantization.
# IQ3_S, IQ4_XS, IQ4_NL do NOT need imatrix.
_IMATRIX_REQUIRED = {
    GGUFQuantType.IQ1_S, GGUFQuantType.IQ1_M,
    GGUFQuantType.IQ2_XXS, GGUFQuantType.IQ2_XS, GGUFQuantType.IQ2_S,
    GGUFQuantType.IQ3_XXS,
}


def _needs_imatrix(qtype: GGUFQuantType) -> bool:
    """Check if a quant type requires an importance matrix."""
    return qtype in _IMATRIX_REQUIRED
