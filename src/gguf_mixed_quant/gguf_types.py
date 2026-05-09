"""GGUF quantization type definitions and utilities."""

from dataclasses import dataclass
from enum import Enum


class GGUFQuantType(Enum):
    """GGUF atomic quantization types which compose all other quants."""

    IQ1_S = "IQ1_S"
    IQ1_M = "IQ1_M"
    IQ2_XXS = "IQ2_XXS"
    IQ2_XS = "IQ2_XS"
    IQ2_S = "IQ2_S"
    Q2_K = "Q2_K"
    IQ3_XXS = "IQ3_XXS"
    IQ3_S = "IQ3_S"
    Q3_K = "Q3_K"
    IQ4_XS = "IQ4_XS"
    IQ4_NL = "IQ4_NL"
    Q4_K = "Q4_K"
    Q5_K = "Q5_K"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"
    F16 = "F16"

    @property
    def ggml_type_name(self) -> str:
        """Return the ggml type name accepted by llama-quantize --tensor-type."""
        return self.value.lower().replace("_k", "_K")


@dataclass(frozen=True)
class GGUFQuantInfo:
    """Metadata for a GGUF quantization type."""

    quant_type: GGUFQuantType
    bits_per_weight: float
    description: str


# Bits-per-weight computed from block_size and bytes_per_block:
#   bpw = bytes_per_block * 8 / block_size
GGUF_QUANT_INFO: dict[GGUFQuantType, GGUFQuantInfo] = {
    GGUFQuantType.IQ1_S: GGUFQuantInfo(GGUFQuantType.IQ1_S, 1.5625, "1-bit importance quants (small)"),
    GGUFQuantType.IQ1_M: GGUFQuantInfo(GGUFQuantType.IQ1_M, 1.75, "1-bit importance quants (medium)"),
    GGUFQuantType.IQ2_XXS: GGUFQuantInfo(GGUFQuantType.IQ2_XXS, 2.0625, "2-bit importance quants (extra-extra-small)"),
    GGUFQuantType.IQ2_XS: GGUFQuantInfo(GGUFQuantType.IQ2_XS, 2.3125, "2-bit importance quants (extra-small)"),
    GGUFQuantType.IQ2_S: GGUFQuantInfo(GGUFQuantType.IQ2_S, 2.5625, "2-bit importance quants (small)"),
    GGUFQuantType.Q2_K: GGUFQuantInfo(GGUFQuantType.Q2_K, 2.625, "2-bit K-quants"),
    GGUFQuantType.IQ3_XXS: GGUFQuantInfo(GGUFQuantType.IQ3_XXS, 3.0625, "3-bit importance quants (extra-extra-small)"),
    GGUFQuantType.IQ3_S: GGUFQuantInfo(GGUFQuantType.IQ3_S, 3.4375, "3-bit importance quants"),
    GGUFQuantType.Q3_K: GGUFQuantInfo(GGUFQuantType.Q3_K, 3.4375, "3-bit K-quants"),
    GGUFQuantType.IQ4_XS: GGUFQuantInfo(GGUFQuantType.IQ4_XS, 4.25, "4-bit importance quants (extra-small)"),
    GGUFQuantType.IQ4_NL: GGUFQuantInfo(GGUFQuantType.IQ4_NL, 4.50, "4-bit importance quants (non-linear)"),
    GGUFQuantType.Q4_K: GGUFQuantInfo(GGUFQuantType.Q4_K, 4.50, "4-bit K-quants"),
    GGUFQuantType.Q5_K: GGUFQuantInfo(GGUFQuantType.Q5_K, 5.50, "5-bit K-quants"),
    GGUFQuantType.Q6_K: GGUFQuantInfo(GGUFQuantType.Q6_K, 6.5625, "6-bit K-quants"),
    GGUFQuantType.Q8_0: GGUFQuantInfo(GGUFQuantType.Q8_0, 8.50, "8-bit"),
    GGUFQuantType.F16: GGUFQuantInfo(GGUFQuantType.F16, 16.0, "16-bit float"),
}


# Ordered list from lowest to highest precision
GGUF_TYPES_BY_PRECISION: list[GGUFQuantType] = sorted(
    GGUF_QUANT_INFO.keys(), key=lambda t: GGUF_QUANT_INFO[t].bits_per_weight
)


def get_bpw(quant_type: GGUFQuantType) -> float:
    """Get bits-per-weight for a quantization type."""
    return GGUF_QUANT_INFO[quant_type].bits_per_weight


def parse_quant_type(name: str) -> GGUFQuantType:
    """Parse a quantization type from a string name."""
    name_upper = name.upper().replace("-", "_")
    try:
        return GGUFQuantType(name_upper)
    except ValueError:
        valid = ", ".join(t.value for t in GGUFQuantType)
        raise ValueError(f"Unknown GGUF quant type: '{name}'. Valid types: {valid}") from None
