"""GGUF quantization type definitions and utilities."""

from dataclasses import dataclass
from enum import Enum


class GGUFQuantType(Enum):
    """GGUF quantization types ordered by bits-per-weight."""

    IQ1_S = "IQ1_S"
    IQ1_M = "IQ1_M"
    IQ2_XXS = "IQ2_XXS"
    IQ2_XS = "IQ2_XS"
    IQ2_S = "IQ2_S"
    IQ2_M = "IQ2_M"
    Q2_K = "Q2_K"
    Q2_K_S = "Q2_K_S"
    IQ3_XXS = "IQ3_XXS"
    IQ3_XS = "IQ3_XS"
    IQ3_S = "IQ3_S"
    IQ3_M = "IQ3_M"
    Q3_K_S = "Q3_K_S"
    Q3_K_M = "Q3_K_M"
    Q3_K_L = "Q3_K_L"
    IQ4_XS = "IQ4_XS"
    IQ4_NL = "IQ4_NL"
    Q4_K_S = "Q4_K_S"
    Q4_K_M = "Q4_K_M"
    Q5_K_S = "Q5_K_S"
    Q5_K_M = "Q5_K_M"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"
    F16 = "F16"

    @property
    def ggml_type_name(self) -> str:
        """Return the ggml type name accepted by llama-quantize --tensor-type."""
        mapping = {
            "IQ1_S": "iq1_s",
            "IQ1_M": "iq1_m",
            "IQ2_XXS": "iq2_xxs",
            "IQ2_XS": "iq2_xs",
            "IQ2_S": "iq2_s",
            "IQ2_M": "iq2_s",  # IQ2_M uses iq2_s as base ggml type
            "Q2_K": "q2_K",
            "Q2_K_S": "q2_K",
            "IQ3_XXS": "iq3_xxs",
            "IQ3_XS": "iq3_s",  # IQ3_XS base type is iq3_s
            "IQ3_S": "iq3_s",
            "IQ3_M": "iq3_s",
            "Q3_K_S": "q3_K",
            "Q3_K_M": "q3_K",
            "Q3_K_L": "q3_K",
            "IQ4_XS": "iq4_xs",
            "IQ4_NL": "iq4_nl",
            "Q4_K_S": "q4_K",
            "Q4_K_M": "q4_K",
            "Q5_K_S": "q5_K",
            "Q5_K_M": "q5_K",
            "Q6_K": "q6_K",
            "Q8_0": "q8_0",
            "F16": "f16",
        }
        return mapping[self.value]


@dataclass(frozen=True)
class GGUFQuantInfo:
    """Metadata for a GGUF quantization type."""

    quant_type: GGUFQuantType
    bits_per_weight: float
    description: str


GGUF_QUANT_INFO: dict[GGUFQuantType, GGUFQuantInfo] = {
    GGUFQuantType.IQ1_S: GGUFQuantInfo(GGUFQuantType.IQ1_S, 1.56, "1-bit importance quants (small)"),
    GGUFQuantType.IQ1_M: GGUFQuantInfo(GGUFQuantType.IQ1_M, 1.75, "1-bit importance quants (medium)"),
    GGUFQuantType.IQ2_XXS: GGUFQuantInfo(GGUFQuantType.IQ2_XXS, 2.06, "2-bit importance quants (extra-extra-small)"),
    GGUFQuantType.IQ2_XS: GGUFQuantInfo(GGUFQuantType.IQ2_XS, 2.31, "2-bit importance quants (extra-small)"),
    GGUFQuantType.IQ2_S: GGUFQuantInfo(GGUFQuantType.IQ2_S, 2.50, "2-bit importance quants (small)"),
    GGUFQuantType.IQ2_M: GGUFQuantInfo(GGUFQuantType.IQ2_M, 2.70, "2-bit importance quants (medium)"),
    GGUFQuantType.Q2_K: GGUFQuantInfo(GGUFQuantType.Q2_K, 2.63, "2-bit K-quants"),
    GGUFQuantType.Q2_K_S: GGUFQuantInfo(GGUFQuantType.Q2_K_S, 2.63, "2-bit K-quants (small)"),
    GGUFQuantType.IQ3_XXS: GGUFQuantInfo(GGUFQuantType.IQ3_XXS, 3.06, "3-bit importance quants (extra-extra-small)"),
    GGUFQuantType.IQ3_XS: GGUFQuantInfo(GGUFQuantType.IQ3_XS, 3.30, "3-bit importance quants (extra-small)"),
    GGUFQuantType.IQ3_S: GGUFQuantInfo(GGUFQuantType.IQ3_S, 3.44, "3-bit importance quants (small)"),
    GGUFQuantType.IQ3_M: GGUFQuantInfo(GGUFQuantType.IQ3_M, 3.70, "3-bit importance quants (medium)"),
    GGUFQuantType.Q3_K_S: GGUFQuantInfo(GGUFQuantType.Q3_K_S, 3.44, "3-bit K-quants (small)"),
    GGUFQuantType.Q3_K_M: GGUFQuantInfo(GGUFQuantType.Q3_K_M, 3.91, "3-bit K-quants (medium)"),
    GGUFQuantType.Q3_K_L: GGUFQuantInfo(GGUFQuantType.Q3_K_L, 4.27, "3-bit K-quants (large)"),
    GGUFQuantType.IQ4_XS: GGUFQuantInfo(GGUFQuantType.IQ4_XS, 4.25, "4-bit importance quants (extra-small)"),
    GGUFQuantType.IQ4_NL: GGUFQuantInfo(GGUFQuantType.IQ4_NL, 4.50, "4-bit importance quants (non-linear)"),
    GGUFQuantType.Q4_K_S: GGUFQuantInfo(GGUFQuantType.Q4_K_S, 4.59, "4-bit K-quants (small)"),
    GGUFQuantType.Q4_K_M: GGUFQuantInfo(GGUFQuantType.Q4_K_M, 4.85, "4-bit K-quants (medium)"),
    GGUFQuantType.Q5_K_S: GGUFQuantInfo(GGUFQuantType.Q5_K_S, 5.54, "5-bit K-quants (small)"),
    GGUFQuantType.Q5_K_M: GGUFQuantInfo(GGUFQuantType.Q5_K_M, 5.69, "5-bit K-quants (medium)"),
    GGUFQuantType.Q6_K: GGUFQuantInfo(GGUFQuantType.Q6_K, 6.56, "6-bit K-quants"),
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
