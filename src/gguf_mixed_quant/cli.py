"""Command-line interface for gguf-mixed-quant."""

import argparse
import sys

from gguf_mixed_quant.sensitivity import compute_sensitivity, list_available_metrics
from gguf_mixed_quant.precision_assignment import assign_gguf_types, assign_gguf_types_multilevel
from gguf_mixed_quant.export import export_overrides
from gguf_mixed_quant.gguf_types import GGUFQuantType


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gguf-mixed-quant",
        description="Mixed-precision GGUF quantization using NNCF sensitivity metrics",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--metric",
        default="weight_quantization_error",
        choices=list(list_available_metrics().keys()),
        help="Sensitivity metric to use (default: weight_quantization_error)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="HuggingFace dataset for data-aware metrics (e.g., 'wikitext')",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=128,
        help="Number of calibration samples (default: 128)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.8,
        help="Fraction of weights for primary (lower) precision (default: 0.8)",
    )
    parser.add_argument(
        "--primary-type",
        default="Q4_K_M",
        help="GGUF type for least-sensitive layers (default: Q4_K_M)",
    )
    parser.add_argument(
        "--backup-type",
        default="Q6_K",
        help="GGUF type for most-sensitive layers (default: Q6_K)",
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=None,
        help="Number of quantization levels for multi-level assignment (overrides primary/backup)",
    )
    parser.add_argument(
        "--quant-types",
        nargs="+",
        default=None,
        help="Explicit quant types for multi-level (from low to high precision)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Quantization group size (default: 128)",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "llama-quantize-args", "table", "gguf"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (prints to stdout if not specified)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for model loading (default: cpu)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Model dtype (default: float32)",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List available sensitivity metrics and exit",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_metrics:
        metrics = list_available_metrics()
        print("Available sensitivity metrics:\n")
        for name, info in metrics.items():
            data_req = "requires calibration data" if info["requires_data"] else "data-free"
            print(f"  {name:<30} ({data_req})")
            print(f"    {info['description']}\n")
        return 0

    import torch

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    # Step 1: Compute sensitivity
    print("=" * 60)
    print("Step 1: Computing sensitivity scores")
    print("=" * 60)

    sensitivity_result = compute_sensitivity(
        model_id=args.model,
        metric=args.metric,
        dataset_name=args.dataset,
        subset_size=args.subset_size,
        group_size=args.group_size,
        torch_dtype=dtype_map[args.dtype],
        device=args.device,
    )

    # Step 2: Assign quantization types
    print("\n" + "=" * 60)
    print("Step 2: Assigning GGUF quantization types")
    print("=" * 60)

    if args.num_levels is not None:
        plan = assign_gguf_types_multilevel(
            sensitivity_result,
            num_levels=args.num_levels,
            quant_types=args.quant_types,
        )
    else:
        plan = assign_gguf_types(
            sensitivity_result,
            ratio=args.ratio,
            primary_type=args.primary_type,
            backup_type=args.backup_type,
        )

    print(f"\n{plan.summary()}")

    # Step 3: Export
    print("\n" + "=" * 60)
    print("Step 3: Exporting results")
    print("=" * 60)

    if args.output_format == "gguf":
        if not args.output:
            print("Error: --output is required for GGUF export", file=sys.stderr)
            return 1

        from gguf_mixed_quant.gguf_export import quantize_to_gguf

        quantize_to_gguf(
            plan=plan,
            output_path=args.output,
            torch_dtype=dtype_map[args.dtype],
        )
    else:
        output = export_overrides(plan, format=args.output_format, output_path=args.output)
        if not args.output:
            print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
