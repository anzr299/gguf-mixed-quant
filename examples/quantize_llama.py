"""Example: Quantize a Llama model with mixed precision GGUF types."""

from pathlib import Path

from gguf_mixed_quant import compute_sensitivity, assign_gguf_types, export_overrides
from gguf_mixed_quant.precision_assignment import (
    assign_gguf_types_multilevel,
    assign_gguf_types_preset,
    refine_baseline,
)
from gguf_mixed_quant.baseline import get_baseline_assignments, baseline_to_map


MODEL_ID = "meta-llama/Llama-3.2-1B"


def example_two_level():
    """Two-level assignment: 80% low precision, 20% high precision."""
    print("=" * 60)
    print("Example 1: Two-level mixed precision (80% Q4_K_M, 20% Q6_K)")
    print("=" * 60)

    scores = compute_sensitivity(
        model_id=MODEL_ID,
        metric="weight_quantization_error",
    )

    plan = assign_gguf_types(
        scores,
        ratio=0.8,
        primary_type="Q4_K_M",
        backup_type="Q6_K",
    )

    print(plan.summary())
    print()

    overrides = export_overrides(plan, format="llama-quantize-args")
    print(overrides[:500])


def example_preset():
    """Preset-based assignment matching llama.cpp tier structure."""
    print("=" * 60)
    print("Example 2: Preset-based Q4_K_M (data-aware)")
    print("=" * 60)

    scores = compute_sensitivity(
        model_id=MODEL_ID,
        metric="max_activation_variance",
        dataset_name="wikitext",
        subset_size=128,
    )

    plan = assign_gguf_types_preset(scores, preset_name="Q4_K_M")
    print(plan.summary())
    print(f"Average BPW: {plan.avg_bpw:.2f}")
    print(f"Distribution: {plan.type_distribution}")


def example_multilevel():
    """Four-level assignment with explicit quant types."""
    print("=" * 60)
    print("Example 3: Four-level mixed precision")
    print("=" * 60)

    scores = compute_sensitivity(
        model_id=MODEL_ID,
        metric="weight_quantization_error",
    )

    plan = assign_gguf_types_multilevel(
        scores,
        num_levels=4,
        quant_types=["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K"],
    )

    print(plan.summary())

    export_overrides(plan, format="json", output_path="mixed_precision_plan.json")
    print("Saved to: mixed_precision_plan.json")


def example_refine(llama_cpp_dir: str, f16_gguf_path: str):
    """Refine llama.cpp's baseline using sensitivity (same file size, better PPL)."""
    print("=" * 60)
    print("Example 4: Refine llama.cpp baseline (recommended)")
    print("=" * 60)

    llama_cpp = Path(llama_cpp_dir)
    f16_gguf = Path(f16_gguf_path)

    # Compute data-aware sensitivity scores
    scores = compute_sensitivity(
        model_id=MODEL_ID,
        metric="max_activation_variance",
        dataset_name="wikitext",
        subset_size=128,
    )

    # Parse llama.cpp's default Q4_K_M assignments
    baseline = get_baseline_assignments(f16_gguf, "Q4_K_M", llama_cpp)
    baseline_map = baseline_to_map(baseline)
    print(f"Baseline: {len(baseline_map)} tensors")

    # Refine: redistribute bump slots by sensitivity
    plan = refine_baseline(
        baseline_map=baseline_map,
        sensitivity_result=scores,
    )
    print(plan.summary())

    # Export overrides for quantizing
    export_overrides(plan, format="llama-quantize-args", output_path="refined_overrides.txt")
    print("\nQuantize with:")
    print(f"  llama-quantize --tensor-type-file refined_overrides.txt "
          f"{f16_gguf} model-refined.gguf Q4_K_M")


def example_refine_with_dip(llama_cpp_dir: str, f16_gguf_path: str):
    """Refine + dip: downgrade insensitive layers, upgrade sensitive ones."""
    print("=" * 60)
    print("Example 5: Refine + dip (downgrade insensitive, promote sensitive)")
    print("=" * 60)

    llama_cpp = Path(llama_cpp_dir)
    f16_gguf = Path(f16_gguf_path)

    scores = compute_sensitivity(
        model_id=MODEL_ID,
        metric="max_activation_variance",
        dataset_name="wikitext",
        subset_size=128,
    )

    baseline = get_baseline_assignments(f16_gguf, "Q4_K_M", llama_cpp)
    baseline_map = baseline_to_map(baseline)

    # dip_fraction=0.1 downgrades 10% of base-type layers one tier lower,
    # then uses the saved bits to promote sensitive layers
    plan = refine_baseline(
        baseline_map=baseline_map,
        sensitivity_result=scores,
        dip_fraction=0.1,
    )
    print(plan.summary())


if __name__ == "__main__":
    # Run data-free examples (no llama.cpp needed)
    example_two_level()
    print()
    example_multilevel()

    # Uncomment below with your paths for data-aware + refine examples:
    # example_preset()
    # example_refine("/tmp/llama_cpp_output/llama.cpp", "/tmp/llama-3.2-1b-f16.gguf")
    # example_refine_with_dip("/tmp/llama_cpp_output/llama.cpp", "/tmp/llama-3.2-1b-f16.gguf")
