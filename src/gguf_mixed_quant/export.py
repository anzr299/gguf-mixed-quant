"""Export mixed-precision plans in various formats."""

import json
from pathlib import Path
from typing import Optional

from gguf_mixed_quant.precision_assignment import MixedPrecisionPlan


def export_overrides(
    plan: MixedPrecisionPlan,
    format: str = "json",
    output_path: Optional[str] = None,
) -> str:
    """
    Export the mixed-precision plan in the specified format.

    :param plan: The quantization plan to export.
    :param format: Output format ('json', 'llama-quantize-args', 'table').
    :param output_path: If provided, write output to this file.
    :return: The formatted output string.
    """
    formatters = {
        "json": _format_json,
        "llama-quantize-args": _format_llama_quantize_args,
        "table": _format_table,
    }

    if format not in formatters:
        valid = ", ".join(formatters.keys())
        raise ValueError(f"Unknown format: '{format}'. Valid formats: {valid}")

    output = formatters[format](plan)

    if output_path:
        Path(output_path).write_text(output, encoding="utf-8")
        print(f"Written to: {output_path}")

    return output


def _format_json(plan: MixedPrecisionPlan) -> str:
    """Export as JSON with layer-to-type mapping."""
    data = {
        "model_id": plan.model_id,
        "metric": plan.metric,
        "avg_bpw": round(plan.avg_bpw, 2),
        "distribution": plan.type_distribution,
        "assignments": [
            {
                "layer": a.layer_name,
                "type": a.quant_type.value,
                "score": round(a.score, 6),
                "bpw": a.bits_per_weight,
            }
            for a in plan.assignments
        ],
    }
    return json.dumps(data, indent=2)


def _format_llama_quantize_args(plan: MixedPrecisionPlan) -> str:
    """
    Export as llama-quantize --override-tensor-type arguments.

    Format: --override-tensor-type REGEX=TYPE
    The regex matches the GGUF tensor name pattern.
    """
    lines = []
    lines.append(f"# Mixed-precision overrides for: {plan.model_id}")
    lines.append(f"# Metric: {plan.metric} | Avg BPW: {plan.avg_bpw:.2f}")
    lines.append("#")
    lines.append("# Usage: llama-quantize model-f16.gguf model-mixed.gguf Q4_K_M \\")

    override_args = []
    for assignment in plan.assignments:
        # Convert HF weight name to GGUF tensor name pattern
        gguf_tensor = _hf_name_to_gguf_pattern(assignment.layer_name)
        override_args.append(f"--override-tensor-type {gguf_tensor}={assignment.quant_type.value}")

    lines.append("#   " + " \\\n#   ".join(override_args))
    lines.append("")
    lines.append("# Override arguments (one per line):")
    lines.extend(override_args)

    return "\n".join(lines)


def _format_table(plan: MixedPrecisionPlan) -> str:
    """Export as a human-readable table."""
    lines = [
        plan.summary(),
        "",
        f"{'Layer':<60} {'Type':<10} {'BPW':<6} {'Score':<12}",
        "-" * 90,
    ]

    for a in sorted(plan.assignments, key=lambda x: x.score):
        name = a.layer_name[:58] if len(a.layer_name) > 58 else a.layer_name
        lines.append(f"{name:<60} {a.quant_type.value:<10} {a.bits_per_weight:<6.2f} {a.score:<12.6f}")

    return "\n".join(lines)


def _hf_name_to_gguf_pattern(hf_name: str) -> str:
    """
    Convert HuggingFace weight name to GGUF tensor name pattern.

    HF: model.layers.0.self_attn.q_proj.weight
    GGUF: blk.0.attn_q.weight

    For simplicity, we use a regex-compatible pattern that matches the
    GGUF tensor naming used by llama.cpp's convert scripts.
    """
    # Common mappings from HF to GGUF naming
    name = hf_name

    # Replace model.layers.N with blk.N
    import re

    name = re.sub(r"model\.layers\.(\d+)", r"blk.\1", name)

    # Replace attention projection names
    replacements = {
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
    }

    for hf_suffix, gguf_suffix in replacements.items():
        if name.endswith(hf_suffix) or name == hf_suffix:
            prefix = name[: -len(hf_suffix)] if name.endswith(hf_suffix) else ""
            name = prefix + gguf_suffix
            break

    return name
