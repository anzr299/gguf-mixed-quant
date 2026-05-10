"""Unsloth Dynamic v2.0 quantization preset configurations.

Reverse-engineered from Unsloth's published GGUF files across 8 dense
architectures (Llama-3.2 1B/3B, Qwen3 4B/8B/14B, Gemma-3 4B/12B/27B).

Each preset defines:
  - base_bits: nominal bit-width of the base quantization level
  - is_iq_base: whether the base type is an I-quant (importance/lookup)
  - top_sentinel: highest-precision type for the most-sensitive tensor(s)
  - bands: ordered list of bit-level bands from lowest to highest, each with:
    - tier: position relative to base (-1, base, +1, +2, ...)
    - bits: nominal bit-width
    - pct: percentage of total weight tensors assigned to this band
    - types: sub-types within the band, ordered by ascending BPW
      Each type entry has:
        - type: GGUFQuantType name
        - bpw: bits-per-weight
        - family: "I" (importance/lookup) or "K" (k-quant SIMD)
        - count: "fixed N" for model-independent counts, "N-M" for range
        - pct_of_band: percentage of tensors within this band
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SubTypeConfig:
    """A single quant type within a band."""

    type_name: str
    bpw: float
    family: str  # "I", "K", or "-" (F16/BF16)
    fixed_count: int | None  # None means variable (model-dependent)
    pct_of_band: float  # approximate % of tensors in this band

    @property
    def is_fixed(self) -> bool:
        return self.fixed_count is not None


@dataclass(frozen=True)
class BandConfig:
    """A bit-level band (tier) in a preset."""

    tier: str  # "-1", "base", "+1", "+2", etc.
    bits: int
    pct_of_total: float
    types: tuple[SubTypeConfig, ...]  # ordered by ascending BPW


@dataclass(frozen=True)
class UnslothPreset:
    """Complete Unsloth Dynamic v2.0 preset configuration."""

    name: str
    base_bits: int
    is_iq_base: bool
    top_sentinel: str  # e.g. "Q5_K", "Q6_K", "Q8_0", "BF16"
    bands: tuple[BandConfig, ...]  # ordered from lowest to highest band


def _t(type_name: str, bpw: float, family: str,
       fixed: int | None = None, pct: float = 100.0) -> SubTypeConfig:
    return SubTypeConfig(type_name, bpw, family, fixed, pct)


def _band(tier: str, bits: int, pct: float,
          *types: SubTypeConfig) -> BandConfig:
    return BandConfig(tier, bits, pct, tuple(types))


# ---------------------------------------------------------------------------
# I-base presets (IQ1_S through IQ3_XXS)
# ---------------------------------------------------------------------------

UD_IQ1_S = UnslothPreset(
    name="UD-IQ1_S",
    base_bits=1,
    is_iq_base=True,
    top_sentinel="Q5_K",
    bands=(
        _band("base", 1, 63,
              _t("IQ1_S", 1.56, "I", pct=67),
              _t("IQ1_M", 1.75, "I", pct=33)),
        _band("+1", 2, 29,
              _t("IQ2_XXS", 2.06, "I", pct=63),
              _t("IQ2_S", 2.50, "I", fixed=5, pct=6),
              _t("Q2_K", 2.63, "K", pct=31)),
        _band("+2", 3, 4,
              _t("IQ3_XXS", 3.06, "I", fixed=5, pct=45),
              _t("IQ3_S", 3.44, "I", fixed=5, pct=45),
              _t("Q3_K", 3.44, "K", fixed=1, pct=10)),
        _band("+3", 4, 12,
              _t("Q4_K", 4.50, "K", pct=100)),
        _band("+4", 5, 0.4,
              _t("Q5_K", 5.50, "K", fixed=1, pct=100)),
    ),
)

UD_IQ1_M = UnslothPreset(
    name="UD-IQ1_M",
    base_bits=1,
    is_iq_base=True,
    top_sentinel="Q5_K",
    bands=(
        _band("base", 1, 59,
              _t("IQ1_S", 1.56, "I", fixed=20, pct=13),
              _t("IQ1_M", 1.75, "I", pct=87)),
        _band("+1", 2, 33,
              _t("IQ2_XXS", 2.06, "I", pct=62),
              _t("IQ2_XS", 2.31, "I", fixed=5, pct=5),
              _t("IQ2_S", 2.50, "I", fixed=5, pct=5),
              _t("Q2_K", 2.63, "K", pct=27)),
        _band("+2", 3, 4,
              _t("IQ3_XXS", 3.06, "I", fixed=5, pct=45),
              _t("IQ3_S", 3.44, "I", fixed=5, pct=45),
              _t("Q3_K", 3.44, "K", fixed=1, pct=10)),
        _band("+3", 4, 12,
              _t("Q4_K", 4.50, "K", pct=100)),
        _band("+4", 5, 0.4,
              _t("Q5_K", 5.50, "K", fixed=1, pct=100)),
    ),
)

UD_IQ2_XXS = UnslothPreset(
    name="UD-IQ2_XXS",
    base_bits=2,
    is_iq_base=True,
    top_sentinel="Q5_K",
    bands=(
        _band("base", 2, 84,
              _t("IQ2_XXS", 2.06, "I", pct=96),
              _t("IQ2_S", 2.50, "I", fixed=5, pct=2),
              _t("Q2_K", 2.63, "K", pct=1)),
        _band("+1", 3, 12,
              _t("IQ3_XXS", 3.06, "I", pct=88),
              _t("IQ3_S", 3.44, "I", fixed=5, pct=10),
              _t("Q3_K", 3.44, "K", fixed=1, pct=2)),
        _band("+2", 4, 12,
              _t("Q4_K", 4.50, "K", pct=100)),
        _band("+3", 5, 0.4,
              _t("Q5_K", 5.50, "K", fixed=1, pct=100)),
    ),
)

UD_IQ2_M = UnslothPreset(
    name="UD-IQ2_M",
    base_bits=2,
    is_iq_base=True,
    top_sentinel="Q5_K",
    bands=(
        _band("base", 2, 61,
              _t("IQ2_XS", 2.31, "I", fixed=5, pct=3),
              _t("IQ2_S", 2.50, "I", pct=97)),
        _band("+1", 3, 34,
              _t("IQ3_XXS", 3.06, "I", fixed=20, pct=22),
              _t("IQ3_S", 3.44, "I", pct=77),
              _t("Q3_K", 3.44, "K", fixed=1, pct=1)),
        _band("+2", 4, 12,
              _t("IQ4_XS", 4.25, "I", pct=3),
              _t("Q4_K", 4.50, "K", pct=97)),
        _band("+3", 5, 0.4,
              _t("Q5_K", 5.50, "K", fixed=1, pct=100)),
    ),
)

UD_IQ3_XXS = UnslothPreset(
    name="UD-IQ3_XXS",
    base_bits=3,
    is_iq_base=True,
    top_sentinel="Q5_K",
    bands=(
        _band("-1", 2, 25,
              _t("IQ2_XS", 2.31, "I", fixed=10, pct=15),
              _t("IQ2_S", 2.50, "I", pct=85)),
        _band("base", 3, 69,
              _t("IQ3_XXS", 3.06, "I", pct=77),
              _t("IQ3_S", 3.44, "I", pct=23),
              _t("Q3_K", 3.44, "K", fixed=1, pct=0.5)),
        _band("+1", 4, 6,
              _t("IQ4_XS", 4.25, "I", fixed=5, pct=14),
              _t("Q4_K", 4.50, "K", pct=86)),
        _band("+2", 5, 0.4,
              _t("Q5_K", 5.50, "K", fixed=1, pct=100)),
    ),
)

# ---------------------------------------------------------------------------
# K-base presets (Q2_K through Q8_0)
# ---------------------------------------------------------------------------

UD_Q2_K_XL = UnslothPreset(
    name="UD-Q2_K_XL",
    base_bits=2,
    is_iq_base=False,
    top_sentinel="Q6_K",
    bands=(
        _band("base", 2, 49,
              _t("IQ2_XS", 2.31, "I", fixed=10, pct=8),
              _t("IQ2_S", 2.50, "I", fixed=10, pct=8),
              _t("Q2_K", 2.63, "K", pct=85)),
        _band("+1", 3, 43,
              _t("IQ3_XXS", 3.06, "I", fixed=10, pct=9),
              _t("IQ3_S", 3.44, "I", fixed=15, pct=13),
              _t("Q3_K", 3.44, "K", pct=78)),
        _band("+2", 4, 8,
              _t("IQ4_XS", 4.25, "I", fixed=5, pct=25),
              _t("Q4_K", 4.50, "K", pct=75)),
        _band("+4", 6, 0.4,
              _t("Q6_K", 6.56, "K", fixed=1, pct=100)),
    ),
)

UD_Q3_K_XL = UnslothPreset(
    name="UD-Q3_K_XL",
    base_bits=3,
    is_iq_base=False,
    top_sentinel="Q6_K",
    bands=(
        _band("base", 3, 49,
              _t("IQ3_XXS", 3.06, "I", fixed=10, pct=8),
              _t("IQ3_S", 3.44, "I", fixed=10, pct=8),
              _t("Q3_K", 3.44, "K", pct=85)),
        _band("+1", 4, 35,
              _t("IQ4_XS", 4.25, "I", fixed=20, pct=22),
              _t("Q4_K", 4.50, "K", pct=78)),
        _band("+2", 5, 14,
              _t("Q5_K", 5.50, "K", pct=100)),
        _band("+3", 6, 2,
              _t("Q6_K", 6.56, "K", pct=100)),
    ),
)

UD_Q4_K_XL = UnslothPreset(
    name="UD-Q4_K_XL",
    base_bits=4,
    is_iq_base=False,
    top_sentinel="Q6_K",
    bands=(
        _band("base", 4, 69,
              _t("IQ4_XS", 4.25, "I", fixed=20, pct=11),
              _t("Q4_K", 4.50, "K", pct=89)),
        _band("+1", 5, 11,
              _t("Q5_K", 5.50, "K", pct=100)),
        _band("+2", 6, 20,
              _t("Q6_K", 6.56, "K", pct=100)),
    ),
)

UD_Q5_K_XL = UnslothPreset(
    name="UD-Q5_K_XL",
    base_bits=5,
    is_iq_base=False,
    top_sentinel="Q8_0",
    bands=(
        _band("-1", 4, 8,
              _t("Q4_K", 4.50, "K", fixed=20, pct=100)),
        _band("base", 5, 65,
              _t("Q5_K", 5.50, "K", pct=100)),
        _band("+1", 6, 27,
              _t("Q6_K", 6.56, "K", pct=100)),
        _band("+2", 8, 0.4,
              _t("Q8_0", 8.50, "K", pct=100)),
    ),
)

UD_Q6_K_XL = UnslothPreset(
    name="UD-Q6_K_XL",
    base_bits=6,
    is_iq_base=False,
    top_sentinel="Q8_0",
    bands=(
        _band("base", 6, 49,
              _t("Q6_K", 6.56, "K", pct=100)),
        _band("+1", 8, 51,
              _t("Q8_0", 8.50, "K", pct=100)),
    ),
)

UD_Q8_K_XL = UnslothPreset(
    name="UD-Q8_K_XL",
    base_bits=8,
    is_iq_base=False,
    top_sentinel="BF16",
    bands=(
        _band("base", 8, 86,
              _t("Q8_0", 8.50, "K", pct=100)),
        _band("+1", 16, 14,
              _t("BF16", 16.0, "-", pct=100)),
    ),
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

UNSLOTH_PRESETS: dict[str, UnslothPreset] = {
    p.name: p for p in [
        UD_IQ1_S, UD_IQ1_M, UD_IQ2_XXS, UD_IQ2_M, UD_IQ3_XXS,
        UD_Q2_K_XL, UD_Q3_K_XL, UD_Q4_K_XL,
        UD_Q5_K_XL, UD_Q6_K_XL, UD_Q8_K_XL,
    ]
}


def get_preset(name: str) -> UnslothPreset:
    """
    Look up an Unsloth Dynamic preset by name.

    :param name: Preset name, e.g. "UD-Q4_K_XL" or "UD-IQ2_M".
    :return: The preset configuration.
    """
    key = name.upper().replace(" ", "")
    if key not in UNSLOTH_PRESETS:
        # Try with UD- prefix
        if not key.startswith("UD-"):
            key = f"UD-{key}"
    if key not in UNSLOTH_PRESETS:
        valid = ", ".join(sorted(UNSLOTH_PRESETS.keys()))
        raise ValueError(f"Unknown preset '{name}'. Available: {valid}")
    return UNSLOTH_PRESETS[key]


def list_presets() -> list[str]:
    """Return all available preset names."""
    return sorted(UNSLOTH_PRESETS.keys())
