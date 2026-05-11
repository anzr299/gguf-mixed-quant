"""Command-line interface for gguf-mixed-quant."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from gguf_mixed_quant.sensitivity import compute_sensitivity, list_available_metrics, list_available_datasets
from gguf_mixed_quant.precision_assignment import assign_gguf_types_preset, two_phase_assign
from gguf_mixed_quant.baseline import get_baseline_assignments, baseline_to_map
from gguf_mixed_quant.export import export_overrides
from gguf_mixed_quant.gguf_types import parse_quant_type


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gguf-mixed-quant",
        description=(
            "Mixed-precision GGUF quantization using NNCF sensitivity metrics.\n\n"
            "Two modes:\n"
            "  Auto mode (default):  --preset Q4_K\n"
            "    Algorithm automatically picks optimal per-layer types.\n\n"
            "  Manual mode:          --preset Q4_K --tiers Q4_K Q5_K Q6_K --tier-ratios 0.65 0.20 0.15\n"
            "    You specify the quant types and what percentage of layers gets each.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        required=False,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--preset",
        required=False,
        help="Base quantization preset (e.g., Q4_K, Q3_K, Q5_K, Q2_K, IQ3_S)",
    )

    # --- Manual mode ---
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=None,
        help="Quant types for each tier, lowest to highest precision "
             "(e.g., Q4_K Q5_K Q6_K). Enables manual mode.",
    )
    parser.add_argument(
        "--tier-ratios",
        nargs="+",
        type=float,
        default=None,
        help="Weight fraction for each tier (must sum to 1.0, e.g., 0.65 0.20 0.15)",
    )

    # --- Sensitivity ---
    parser.add_argument(
        "--metric",
        default="max_activation_variance",
        choices=list(list_available_metrics().keys()),
        help="Sensitivity metric (default: max_activation_variance)",
    )
    parser.add_argument(
        "--dataset",
        default="wikitext",
        help="Calibration dataset: 'wikitext' (default), 'nemotron', 'reasoning', "
             "'coding', or any HF dataset name",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Number of calibration samples (default: per-dataset, fallback 128)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Max sequence length for tokenization (default: per-dataset)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Quantization group size (default: 128)",
    )

    # --- Model loading ---
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

    # --- llama.cpp pipeline ---
    parser.add_argument(
        "--llama-cpp",
        default=None,
        help="Path to llama.cpp directory (auto-detected if not set)",
    )
    parser.add_argument(
        "--f16-gguf",
        default=None,
        help="Path to existing F16 GGUF file (skip conversion step)",
    )
    parser.add_argument(
        "--imatrix",
        default=None,
        help="Path to importance matrix file (.dat) for IQ quantization types",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output GGUF file path (default: <model>-mixed.gguf)",
    )

    # --- Auto mode tuning ---
    parser.add_argument(
        "--extra-bpw",
        type=float,
        default=0.0,
        help="Extra avg bits-per-weight above baseline (0.0 = same size). Auto mode only.",
    )
    parser.add_argument(
        "--adaptive-bands",
        action="store_true",
        default=False,
        help="Scale band ratios by sensitivity spread (clustered scores → larger base band). Auto mode only.",
    )

    # --- Info ---
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

    return parser.parse_args(argv)


def _resolve_model_path(model_id: str) -> Path:
    """Resolve a HuggingFace model ID to its local snapshot directory."""
    local = Path(model_id)
    if local.is_dir():
        return local

    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(model_id))
    except Exception:
        pass

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
        hint_path = Path(hint)
        if hint_path.exists():
            return hint_path

    quantize_bin = shutil.which("llama-quantize")
    if quantize_bin:
        return Path(quantize_bin).parent.parent.parent

    candidates = [
        Path("/tmp/llama_cpp_output/llama.cpp"),
        Path.home() / "llama.cpp",
        Path("/usr/local/share/llama.cpp"),
    ]
    for candidate in candidates:
        if (candidate / "build" / "bin" / "llama-quantize").exists():
            return candidate

    return None


def _get_f16_gguf(args, llama_cpp: Path) -> Path | None:
    """Get or create the F16 GGUF file for the model."""
    if args.f16_gguf:
        f16_path = Path(args.f16_gguf)
        if not f16_path.exists():
            print(f"Error: F16 GGUF not found: {f16_path}", file=sys.stderr)
            return None
        return f16_path

    model_name = args.model.split("/")[-1].lower()
    cache_candidates = [
        Path(f"/tmp/{model_name}-f16.gguf"),
        Path(f"/tmp/llama_cpp_output/{model_name}-f16.gguf"),
    ]
    for candidate in cache_candidates:
        if candidate.exists():
            print(f"  Using cached F16 GGUF: {candidate}")
            return candidate

    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"Error: convert_hf_to_gguf.py not found at {convert_script}", file=sys.stderr)
        return None

    f16_gguf = Path(f"/tmp/{model_name}-f16.gguf")
    print(f"\nConverting {args.model} -> F16 GGUF...")
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


def _run_quantize_pipeline(args, plan, baseline_map: dict[str, str] | None = None) -> int:
    """Run the full convert -> quantize pipeline using llama.cpp."""
    llama_cpp = _find_llama_cpp(args.llama_cpp)
    if llama_cpp is None:
        print("Error: Cannot find llama.cpp. Pass --llama-cpp /path/to/llama.cpp", file=sys.stderr)
        return 1

    quantize_bin = llama_cpp / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        print(f"Error: llama-quantize not found at {quantize_bin}", file=sys.stderr)
        return 1

    output_path = args.output or f"{args.model.split('/')[-1]}-mixed.gguf"
    base_quant = args.preset or "Q4_K"

    f16_gguf = _get_f16_gguf(args, llama_cpp)
    if f16_gguf is None:
        return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tensor_types_file = tmpdir / "tensor_types.txt"

        overrides = export_overrides(plan, format="llama-quantize-args")
        lines = [line for line in overrides.split("\n") if not line.startswith("#") and "=" in line]

        # Pass unscored tensors through at their baseline type
        if baseline_map:
            scored_names = {line.split("=")[0] for line in lines}
            _LARGE_TENSORS = {"token_embd.weight", "output.weight"}
            _F32_TYPES = {"f32", "f16"}
            override_count = 0
            for gguf_name, ggml_type in baseline_map.items():
                if (
                    gguf_name not in scored_names
                    and gguf_name not in _LARGE_TENSORS
                    and ggml_type.lower() not in _F32_TYPES
                ):

                    lines.append(f"{gguf_name}={ggml_type}")
                    override_count += 1
            if override_count > 0:
                print(f"  Passing {override_count} unscored tensors through at baseline type")

        tensor_types_file.write_text("\n".join(lines), encoding="utf-8")

        print(f"\nQuantizing with mixed precision -> {output_path}")
        quant_cmd = [
            str(quantize_bin), "--tensor-type-file", str(tensor_types_file),
        ]
        if args.imatrix:
            quant_cmd.extend(["--imatrix", str(args.imatrix)])
        quant_cmd.extend([str(f16_gguf), output_path, base_quant])
        result = subprocess.run(
            quant_cmd,
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"Error quantizing:\n{result.stderr}", file=sys.stderr)
            return 1

        for line in result.stdout.split("\n"):
            if "quant size" in line:
                print(f"  {line.strip()}")

    print(f"\nDone! Output: {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # --- Info modes ---
    if args.list_metrics:
        metrics = list_available_metrics()
        print("Available sensitivity metrics:\n")
        for name, info in metrics.items():
            data_req = "requires calibration data" if info["requires_data"] else "data-free"
            print(f"  {name:<30} ({data_req})")
            print(f"    {info['description']}\n")
        return 0

    if args.list_presets:
        from gguf_mixed_quant.gguf_types import GGUFQuantType
        print("Supported base presets (pass to --preset):\n")
        for t in GGUFQuantType:
            if t != GGUFQuantType.F16:
                print(f"  {t.value}")
        return 0

    if args.list_datasets:
        datasets = list_available_datasets()
        print("Available named datasets:\n")
        for name, desc in datasets.items():
            print(f"  {name:<12} {desc}")
        print("\n  (You can also pass any HuggingFace dataset name directly)")
        return 0

    # --- Validation ---
    if not args.model:
        print("Error: --model is required", file=sys.stderr)
        return 1

    if not args.preset:
        print("Error: --preset is required (e.g., --preset Q4_K)", file=sys.stderr)
        return 1

    is_manual = args.tiers is not None or args.tier_ratios is not None
    if is_manual and (args.tiers is None or args.tier_ratios is None):
        print("Error: --tiers and --tier-ratios must both be specified for manual mode",
              file=sys.stderr)
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
        seq_len=args.seq_len,
        group_size=args.group_size,
        torch_dtype=dtype_map[args.dtype],
        device=args.device,
    )

    # Step 2: Assign quantization types
    print("\n" + "=" * 60)
    print("Step 2: Assigning GGUF quantization types")
    print("=" * 60)

    if is_manual:
        # Manual mode: user specifies tiers and ratios
        tier_types = [parse_quant_type(t) for t in args.tiers]
        plan = assign_gguf_types_preset(
            sensitivity_result,
            tiers=tier_types,
            ratios=args.tier_ratios,
        )
    else:
        # Auto mode: algorithm picks optimal per-layer types
        plan = None  # computed after baseline

    # Common pipeline: resolve llama.cpp, F16 GGUF, and baseline
    llama_cpp = _find_llama_cpp(args.llama_cpp)
    if llama_cpp is None:
        print("Error: Cannot find llama.cpp. Pass --llama-cpp /path/to/llama.cpp",
              file=sys.stderr)
        return 1

    f16_gguf = _get_f16_gguf(args, llama_cpp)
    if f16_gguf is None:
        return 1

    print(f"  Parsing llama.cpp baseline for {args.preset}...")
    imatrix_path = Path(args.imatrix) if args.imatrix else None
    baseline_assignments = get_baseline_assignments(
        f16_gguf, args.preset, llama_cpp, imatrix_path=imatrix_path,
    )
    baseline_map = baseline_to_map(baseline_assignments)

    if plan is None:
        # Auto mode: run two-phase assignment with baseline
        print(f"  Baseline: {len(baseline_map)} tensors assigned")
        plan = two_phase_assign(
            baseline_map=baseline_map,
            sensitivity_result=sensitivity_result,
            extra_bpw=args.extra_bpw,
            has_imatrix=bool(args.imatrix),
            adaptive_bands=args.adaptive_bands,
        )

    print(f"\n{plan.summary()}")
    return _run_quantize_pipeline(args, plan, baseline_map=baseline_map)
