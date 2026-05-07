"""Example: Quantize a Llama model with mixed precision GGUF types."""

from gguf_mixed_quant import compute_sensitivity, assign_gguf_types, export_overrides
from gguf_mixed_quant.precision_assignment import assign_gguf_types_multilevel


def main():
    # --- Example 1: Two-level assignment (like NNCF's ratio approach) ---
    print("=" * 60)
    print("Example 1: Two-level mixed precision (80% Q4_K_M, 20% Q6_K)")
    print("=" * 60)

    # Compute sensitivity scores (data-free, no calibration data needed)
    scores = compute_sensitivity(
        model_id="meta-llama/Llama-3.2-1B",
        metric="weight_quantization_error",
    )

    # Assign GGUF types: 80% least-sensitive -> Q4_K_M, rest -> Q6_K
    plan = assign_gguf_types(
        scores,
        ratio=0.8,
        primary_type="Q4_K_M",
        backup_type="Q6_K",
    )

    print(plan.summary())
    print()

    # Export as llama-quantize CLI args
    overrides = export_overrides(plan, format="llama-quantize-args")
    print(overrides[:500])

    # --- Example 2: Multi-level assignment ---
    print("\n" + "=" * 60)
    print("Example 2: Four-level mixed precision")
    print("=" * 60)

    plan_multi = assign_gguf_types_multilevel(
        scores,
        num_levels=4,
        quant_types=["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K"],
    )

    print(plan_multi.summary())
    print()

    # Export as JSON
    json_output = export_overrides(plan_multi, format="json", output_path="mixed_precision_plan.json")
    print("Saved to: mixed_precision_plan.json")


def example_data_aware():
    """Example with data-aware metric (requires calibration data)."""
    print("=" * 60)
    print("Data-aware example: HAWQ metric with wikitext calibration")
    print("=" * 60)

    scores = compute_sensitivity(
        model_id="meta-llama/Llama-3.2-1B",
        metric="hessian_input_activation",
        dataset_name="wikitext",
        subset_size=64,
    )

    plan = assign_gguf_types(
        scores,
        ratio=0.8,
        primary_type="Q4_K_S",
        backup_type="Q8_0",
    )

    print(plan.summary())

    # Save overrides for use with llama-quantize
    export_overrides(plan, format="llama-quantize-args", output_path="llama_quantize_overrides.txt")
    print("\nRun with llama-quantize:")
    print("  llama-quantize model-f16.gguf model-mixed.gguf Q4_K_M $(cat llama_quantize_overrides.txt)")


if __name__ == "__main__":
    main()
    # Uncomment to run data-aware example:
    # example_data_aware()
