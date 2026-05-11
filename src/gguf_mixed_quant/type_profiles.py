"""Bit-level groupings for GGUF quantization types."""

from dataclasses import dataclass

from gguf_mixed_quant.gguf_types import GGUFQuantType, get_bpw


@dataclass(frozen=True)
class BitLevel:
    """Quant types sharing the same nominal bit width."""

    nominal_bits: int
    variants: tuple[GGUFQuantType, ...]  # ordered low→high bpw


BIT_LEVELS: list[BitLevel] = [
    BitLevel(1, (GGUFQuantType.IQ1_S, GGUFQuantType.IQ1_M)),
    BitLevel(2, (GGUFQuantType.IQ2_XXS, GGUFQuantType.IQ2_XS, GGUFQuantType.IQ2_S, GGUFQuantType.Q2_K)),
    BitLevel(3, (GGUFQuantType.IQ3_XXS, GGUFQuantType.IQ3_S, GGUFQuantType.Q3_K)),
    BitLevel(4, (GGUFQuantType.IQ4_XS, GGUFQuantType.IQ4_NL, GGUFQuantType.Q4_K)),
    BitLevel(5, (GGUFQuantType.Q5_K,)),
    BitLevel(6, (GGUFQuantType.Q6_K,)),
    BitLevel(8, (GGUFQuantType.Q8_0,)),
]

BIT_LEVEL_MAP: dict[int, BitLevel] = {bl.nominal_bits: bl for bl in BIT_LEVELS}


def get_bit_level_for_type(qtype: GGUFQuantType) -> BitLevel:
    """Find which bit level a quant type belongs to."""
    for bl in BIT_LEVELS:
        if qtype in bl.variants:
            return bl
    raise ValueError(f"{qtype} not in any bit level")
