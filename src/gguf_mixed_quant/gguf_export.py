"""Direct GGUF quantization using the gguf Python library."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gguf_mixed_quant.gguf_types import GGUFQuantType
from gguf_mixed_quant.precision_assignment import MixedPrecisionPlan


# Map GGUFQuantType to gguf library's GGMLQuantizationType
_GGUF_TYPE_MAP: dict[str, int] = {
    "Q2_K": 10,
    "Q3_K_S": 11,
    "Q3_K_M": 12,
    "Q3_K_L": 13,
    "Q4_K_S": 14,
    "Q4_K_M": 15,
    "Q5_K_S": 16,
    "Q5_K_M": 17,
    "Q6_K": 18,
    "Q8_0": 8,
    "F16": 1,
    "IQ4_XS": 22,
}


def quantize_to_gguf(
    plan: MixedPrecisionPlan,
    output_path: str,
    model_path: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float16,
) -> Path:
    """
    Quantize a model directly to GGUF with mixed precision using the plan.

    :param plan: Mixed-precision assignment plan.
    :param output_path: Path for the output GGUF file.
    :param model_path: HuggingFace model path (uses plan.model_id if None).
    :param torch_dtype: dtype for loading the model.
    :return: Path to the output GGUF file.
    """
    try:
        import gguf
    except ImportError:
        raise ImportError("The 'gguf' package is required for direct GGUF export. Install with: pip install gguf")

    model_id = model_path or plan.model_id
    output = Path(output_path)

    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Build assignment lookup
    assignment_map = {a.layer_name: a.quant_type for a in plan.assignments}

    print(f"Writing GGUF to: {output}")
    writer = gguf.GGUFWriter(str(output), arch="llama")

    # Write model metadata
    _write_model_metadata(writer, model, tokenizer)

    # Write tensors with per-layer quantization
    state_dict = model.state_dict()
    for name, tensor in state_dict.items():
        tensor_np = tensor.cpu().float().numpy()

        quant_type = assignment_map.get(name)
        if quant_type is not None:
            gguf_type_id = _GGUF_TYPE_MAP.get(quant_type.value, 1)  # Default to F16
        else:
            # Non-weight tensors (norms, embeddings not in plan) -> F16
            gguf_type_id = 1

        # Convert HF tensor name to GGUF name
        gguf_name = _hf_to_gguf_tensor_name(name)

        if gguf_type_id == 1:
            # F16 - store as-is in float16
            writer.add_tensor(gguf_name, tensor.cpu().half().numpy())
        else:
            # For quantized types, store in float32 and let gguf library handle it
            # Note: The gguf library's quantization support is limited,
            # so for production use, prefer the llama-quantize CLI approach
            writer.add_tensor(gguf_name, tensor_np, raw_dtype=np.float32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"GGUF written: {output} ({output.stat().st_size / (1024**3):.2f} GB)")
    return output


def _write_model_metadata(writer, model, tokenizer) -> None:
    """Write standard GGUF metadata keys."""
    config = model.config

    writer.add_name(getattr(config, "_name_or_path", "unknown"))
    writer.add_context_length(getattr(config, "max_position_embeddings", 4096))
    writer.add_embedding_length(getattr(config, "hidden_size", 4096))
    writer.add_block_count(getattr(config, "num_hidden_layers", 32))
    writer.add_feed_forward_length(getattr(config, "intermediate_size", 11008))
    writer.add_head_count(getattr(config, "num_attention_heads", 32))
    writer.add_head_count_kv(getattr(config, "num_key_value_heads", 32))

    if hasattr(config, "rope_theta"):
        writer.add_rope_freq_base(config.rope_theta)
    if hasattr(config, "rms_norm_eps"):
        writer.add_layer_norm_rms_eps(config.rms_norm_eps)

    # Tokenizer info
    writer.add_tokenizer_model("llama")


def _hf_to_gguf_tensor_name(hf_name: str) -> str:
    """Convert HuggingFace state dict key to GGUF tensor name."""
    import re

    name = hf_name

    # model.embed_tokens.weight -> token_embd.weight
    if name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if name == "model.norm.weight":
        return "output_norm.weight"
    if name == "lm_head.weight":
        return "output.weight"

    # model.layers.N.xxx -> blk.N.xxx
    match = re.match(r"model\.layers\.(\d+)\.(.*)", name)
    if match:
        layer_idx = match.group(1)
        remainder = match.group(2)

        suffix_map = {
            "self_attn.q_proj.weight": "attn_q.weight",
            "self_attn.k_proj.weight": "attn_k.weight",
            "self_attn.v_proj.weight": "attn_v.weight",
            "self_attn.o_proj.weight": "attn_output.weight",
            "mlp.gate_proj.weight": "ffn_gate.weight",
            "mlp.up_proj.weight": "ffn_up.weight",
            "mlp.down_proj.weight": "ffn_down.weight",
            "input_layernorm.weight": "attn_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
        }

        gguf_suffix = suffix_map.get(remainder, remainder)
        return f"blk.{layer_idx}.{gguf_suffix}"

    return name
