"""Command-line interface for gguf-mixed-quant."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from gguf_mixed_quant.sensitivity import compute_sensitivity, list_available_metrics, list_available_datasets
from gguf_mixed_quant.precision_assignment import (
    assign_gguf_types,
    assign_gguf_types_multilevel,
    assign_gguf_types_preset,
    refine_baseline,
    list_presets,
    PRESETS,
)
from gguf_mixed_quant.baseline import get_baseline_assignments, baseline_to_map
from gguf_mixed_quant.export import export_overrides
from gguf_mixed_quant.gguf_types import GGUFQuantType


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gguf-mixed-quant",
        description="Mixed-precision GGUF quantization using NNCF sensitivity metrics",
    )

    parser.add_argument(
        "--model",
        default=None,
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
        help="Dataset for data-aware metrics: 'wikitext', 'reasoning' (gsm8k), "
             "'coding' (github-code), 'contextual' (LongBench), or any HF dataset name",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=128,
        help="Number of calibration samples (default: 128)",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Multi-level quantization preset (e.g., Q4_K_M, Q3_K_M, Q5_K_M, Q2_K, Q3_K_L, Q4_K_S). "
             "Overrides --ratio/--primary-type/--backup-type/--num-levels.",
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
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available quantization presets and exit",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available named datasets and exit",
    )
    parser.add_argument(
        "--llama-cpp",
        default=None,
        help="Path to llama.cpp directory (for --quantize pipeline). Auto-detected if not set.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Run full pipeline: compute scores → convert to GGUF → quantize with mixed precision. "
             "Requires llama.cpp. Output is a quantized GGUF file (set with --output).",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Refine llama.cpp's baseline assignments instead of assigning from scratch. "
             "Parses llama-quantize's per-tensor decisions and swaps types based on "
             "sensitivity scores. Implies --quantize. Requires llama.cpp and --preset.",
    )
    parser.add_argument(
        "--swap-count",
        type=int,
        default=None,
        help="Number of layer swaps when using --refine (default: half of bumped layers)",
    )
    parser.add_argument(
        "--dip-fraction",
        type=float,
        default=0.0,
        help="Fraction of base-type weights to downgrade one tier (0.0–1.0). "
             "Saved bits fund upgrades for sensitive layers. Try 0.1 (default: 0.0, disabled).",
    )
    parser.add_argument(
        "--f16-gguf",
        default=None,
        help="Path to existing F16 GGUF file (skip conversion step)",
    )

    return parser.parse_args(argv)


def _resolve_model_path(model_id: str) -> Path:
    """Resolve a HuggingFace model ID to its local snapshot directory."""
    local = Path(model_id)
    if local.is_dir():
        return local

    # Try huggingface_hub snapshot download path
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(model_id))
    except Exception:
        pass

    # Fallback: check cache manually
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_dir / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if repo_dir.exists():
        snapshots = list(repo_dir.iterdir())
        if snapshots:
            return snapshots[-1]

    raise FileNotFoundError(f"Cannot resolve model path for: {model_id}")


def _find_llama_cpp(hint: str | None = None) -> Path | None:
    """Find llama.cpp directory by checking common locations."""
    if hint:
        p = Path(hint)
        if p.exists():
            return p

    # Check PATH for llama-quantize
    quantize_bin = shutil.which("llama-quantize")
    if quantize_bin:
        # e.g. /path/to/llama.cpp/build/bin/llama-quantize -> /path/to/llama.cpp
        return Path(quantize_bin).parent.parent.parent

    # Common locations
    candidates = [
        Path("/tmp/llama_cpp_output/llama.cpp"),
        Path.home() / "llama.cpp",
        Path("/usr/local/share/llama.cpp"),
    ]
    for c in candidates:
        if (c / "build" / "bin" / "llama-quantize").exists():
            return c

    return None


def _get_f16_gguf(args, llama_cpp: Path) -> Path | None:
    """Get or create the F16 GGUF file for the model."""
    if args.f16_gguf:
        f16 = Path(args.f16_gguf)
        if not f16.exists():
            print(f"Error: F16 GGUF not found: {f16}", file=sys.stderr)
            return None
        return f16

    # Check common cache locations
    model_name = args.model.split("/")[-1].lower()
    cache_candidates = [
        Path(f"/tmp/{model_name}-f16.gguf"),
        Path(f"/tmp/llama_cpp_output/{model_name}-f16.gguf"),
    ]
    for c in cache_candidates:
        if c.exists():
            print(f"  Using cached F16 GGUF: {c}")
            return c

    # Convert HF → F16 GGUF
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"Error: convert_hf_to_gguf.py not found at {convert_script}", file=sys.stderr)
        return None

    f16_gguf = Path(f"/tmp/{model_name}-f16.gguf")
    print(f"\nConverting {args.model} → F16 GGUF...")
    model_path = _resolve_model_path(args.model)
    result = subprocess.run(
        [sys.executable, str(convert_script), str(model_path),
         "--outfile", str(f16_gguf), "--outtype", "f16"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error converting to GGUF:\n{result.stderr}", file=sys.stderr)
        return None
    print(f"  F16 GGUF: {f16_gguf.stat().st_size / (1024**3):.2f} GB")
    return f16_gguf


def _run_quantize_pipeline(args, plan) -> int:
    """Run the full convert → quantize pipeline using llama.cpp."""
    llama_cpp = _find_llama_cpp(args.llama_cpp)
    if llama_cpp is None:
        print("Error: Cannot find llama.cpp. Pass --llama-cpp /path/to/llama.cpp", file=sys.stderr)
        return 1

    quantize_bin = llama_cpp / "build" / "bin" / "llama-quantize"

    if not quantize_bin.exists():
        print(f"Error: llama-quantize not found at {quantize_bin}", file=sys.stderr)
        return 1

    output_path = args.output or f"{args.model.split('/')[-1]}-mixed.gguf"
    base_quant = args.preset or "Q4_K_M"

    # Get F16 GGUF (reuse if already created by --refine)
    f16_gguf = _get_f16_gguf(args, llama_cpp)
    if f16_gguf is None:
        return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tensor_types_file = tmpdir / "tensor_types.txt"

        # Write tensor type overrides
        overrides = export_overrides(plan, format="llama-quantize-args")
        lines = [line for line in overrides.split("\n") if not line.startswith("#") and "=" in line]
        tensor_types_file.write_text("\n".join(lines), encoding="utf-8")

        # Quantize with mixed precision
        print(f"\nQuantizing with mixed precision → {output_path}")
        result = subprocess.run(
            [str(quantize_bin), "--tensor-type-file", str(tensor_types_file),
             str(f16_gguf), output_path, base_quant],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"Error quantizing:\n{result.stderr}", file=sys.stderr)
            return 1

        # Extract final size info from output
        for line in result.stdout.split("\n"):
            if "quant size" in line:
                print(f"  {line.strip()}")

    print(f"\nDone! Output: {output_path}")
    return 0

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

    if args.list_presets:
        presets = list_presets()
        print("Available quantization presets (matching llama.cpp):\n")
        for name, desc in presets.items():
            preset = PRESETS[name]
            tiers_str = " → ".join(t.value for t in preset.tiers)
            print(f"  {name:<10} {desc}")
            print(f"             tiers: {tiers_str}")
            print(f"             ratios: {preset.ratios}\n")
        return 0

    if args.list_datasets:
        from gguf_mixed_quant.sensitivity import list_available_datasets
        datasets = list_available_datasets()
        print("Available named datasets:\n")
        for name, desc in datasets.items():
            print(f"  {name:<12} {desc}")
        print("\n  (You can also pass any HuggingFace dataset name directly)")
        return 0

    if not args.model:
        print("Error: --model is required", file=sys.stderr)
        return 1

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

    if args.refine:
        # Refine mode: parse llama.cpp baseline and swap based on sensitivity
        preset_name = args.preset or "Q4_K_M"
        llama_cpp = _find_llama_cpp(args.llama_cpp)
        if llama_cpp is None:
            print("Error: --refine requires llama.cpp. Pass --llama-cpp /path/to/llama.cpp", file=sys.stderr)
            return 1

        # Get or convert F16 GGUF
        f16_gguf = _get_f16_gguf(args, llama_cpp)
        if f16_gguf is None:
            return 1

        print(f"  Parsing llama.cpp baseline for {preset_name}...")
        baseline_assignments = get_baseline_assignments(f16_gguf, preset_name, llama_cpp)
        baseline_map = baseline_to_map(baseline_assignments)
        print(f"  Baseline: {len(baseline_map)} tensors assigned")

        plan = refine_baseline(
            baseline_map=baseline_map,
            sensitivity_result=sensitivity_result,
            swap_count=args.swap_count,
            dip_fraction=args.dip_fraction,
        )
    elif args.preset is not None:
        plan = assign_gguf_types_preset(
            sensitivity_result,
            preset_name=args.preset,
        )
    elif args.num_levels is not None:
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

    # Step 3: Export or run pipeline
    if args.quantize or args.refine:
        return _run_quantize_pipeline(args, plan)

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
