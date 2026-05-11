"""Export mixed-precision plans to llama-quantize format."""

import json
from pathlib import Path

from gguf_mixed_quant.precision_assignment import MixedPrecisionPlan, _hf_to_gguf_name


def export_overrides(
    plan: MixedPrecisionPlan,
    format: str = "json",
    output_path: str | None = None,
) -> str:
    """
    Export the plan in the specified format.

    :param plan: Quantization plan to export.
    :param format: 'json', 'llama-quantize-args', or 'table'.
    :param output_path: If set, write output to this file.
    :return: Formatted output string.
    """
    formatters = {
        "json": _format_json,
        "llama-quantize-args": _format_llama_quantize_args,
        "table": _format_table,
    }
    if format not in formatters:
        valid = ", ".join(formatters.keys())
        raise ValueError(f"Unknown format: '{format}'. Valid: {valid}")

    output = formatters[format](plan)
    if output_path:
        Path(output_path).write_text(output, encoding="utf-8")
    return output


def _format_json(plan: MixedPrecisionPlan) -> str:
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
    """Format as tensor-type-file lines: GGUF_NAME=ggml_type."""
    lines = [
        f"# {plan.model_id} | {plan.metric} | BPW {plan.avg_bpw:.2f}",
    ]
    for a in plan.assignments:
        gguf_name = _hf_to_gguf_name(a.layer_name)
        lines.append(f"{gguf_name}={a.quant_type.ggml_name}")
    return "\n".join(lines)


def _format_table(plan: MixedPrecisionPlan) -> str:
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
