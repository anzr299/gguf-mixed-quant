"""GGUF quantization type definitions."""

from enum import Enum


class GGUFQuantType(Enum):
    """GGUF per-tensor quantization types supported by llama.cpp."""

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
    F32 = "F32"

    @property
    def ggml_name(self) -> str:
        """The ggml type string accepted by llama-quantize."""
        return self.value.lower().replace("_k", "_K")


# Bits-per-weight for each type (from ggml block_size / bytes_per_block).
_BPW: dict[GGUFQuantType, float] = {
    GGUFQuantType.IQ1_S: 1.5625,
    GGUFQuantType.IQ1_M: 1.75,
    GGUFQuantType.IQ2_XXS: 2.0625,
    GGUFQuantType.IQ2_XS: 2.3125,
    GGUFQuantType.IQ2_S: 2.5625,
    GGUFQuantType.Q2_K: 2.625,
    GGUFQuantType.IQ3_XXS: 3.0625,
    GGUFQuantType.IQ3_S: 3.4375,
    GGUFQuantType.Q3_K: 3.4375,
    GGUFQuantType.IQ4_XS: 4.25,
    GGUFQuantType.IQ4_NL: 4.50,
    GGUFQuantType.Q4_K: 4.50,
    GGUFQuantType.Q5_K: 5.50,
    GGUFQuantType.Q6_K: 6.5625,
    GGUFQuantType.Q8_0: 8.50,
    GGUFQuantType.F16: 16.0,
    GGUFQuantType.F32: 32.0,
}


def get_bpw(qtype: GGUFQuantType) -> float:
    """Bits-per-weight for a quantization type."""
    return _BPW[qtype]


def is_iq_type(qtype: GGUFQuantType) -> bool:
    """True if type uses importance-matrix (IQ) dequantization."""
    return qtype.value.startswith("IQ")


def parse_quant_type(name: str) -> GGUFQuantType:
    """Parse a quant type from string (case-insensitive)."""
    upper = name.upper().replace("-", "_")
    try:
        return GGUFQuantType(upper)
    except ValueError:
        valid = ", ".join(t.value for t in GGUFQuantType)
        raise ValueError(f"Unknown quant type: '{name}'. Valid: {valid}") from None
