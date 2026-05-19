"""Test two_phase_assign with all presets against synthetic Qwen3.5-0.6B layer data."""

import random

from gguf_mixed_quant.gguf_types import GGUFQuantType, get_bpw
from gguf_mixed_quant.precision_assignment import (
    GGML_TO_QUANT_TYPE,
    two_phase_assign,
    _hf_to_gguf_name,
)
from gguf_mixed_quant.sensitivity import LayerSensitivity, SensitivityResult

# Qwen3-0.6B: 28 layers, hidden=1024, intermediate=3072, kv_heads=4, q_heads=16
NUM_LAYERS = 28
HIDDEN = 1024
INTERMEDIATE = 3072
KV_HEADS = 4
Q_HEADS = 16
HEAD_DIM = HIDDEN // Q_HEADS  # 64

# Weight shapes per layer
LAYER_WEIGHTS = {
    "self_attn.q_proj.weight": Q_HEADS * HEAD_DIM * HIDDEN,      # 1024*1024
    "self_attn.k_proj.weight": KV_HEADS * HEAD_DIM * HIDDEN,     # 256*1024
    "self_attn.v_proj.weight": KV_HEADS * HEAD_DIM * HIDDEN,     # 256*1024
    "self_attn.o_proj.weight": HIDDEN * Q_HEADS * HEAD_DIM,      # 1024*1024
    "mlp.gate_proj.weight": INTERMEDIATE * HIDDEN,                # 3072*1024
    "mlp.up_proj.weight": INTERMEDIATE * HIDDEN,                  # 3072*1024
    "mlp.down_proj.weight": HIDDEN * INTERMEDIATE,                # 1024*3072
}

# Presets to test: all K-base and IQ-base types
PRESETS: list[GGUFQuantType] = [
    GGUFQuantType.Q2_K,
    GGUFQuantType.Q3_K,
    GGUFQuantType.Q4_K,
    GGUFQuantType.Q5_K,
    GGUFQuantType.IQ2_XXS,
    GGUFQuantType.IQ2_XS,
    GGUFQuantType.IQ2_S,
    GGUFQuantType.IQ3_XXS,
    GGUFQuantType.IQ3_S,
    GGUFQuantType.IQ4_XS,
    GGUFQuantType.IQ4_NL,
]


def _make_layers() -> list[LayerSensitivity]:
    """Build synthetic layer list with realistic score distribution."""
    random.seed(42)
    layers: list[LayerSensitivity] = []
    for i in range(NUM_LAYERS):
        for suffix, nweights in LAYER_WEIGHTS.items():
            name = f"model.layers.{i}.{suffix}"
            # Score: first/last layers more sensitive, MLP less than attention
            layer_pos = abs(i - NUM_LAYERS / 2) / (NUM_LAYERS / 2)
            is_attn = "attn" in suffix
            base_score = 0.2 + 0.6 * layer_pos + (0.15 if is_attn else 0.0)
            score = base_score + random.gauss(0, 0.05)
            vr = max(1.0, random.gauss(2.0, 0.8))
            layers.append(LayerSensitivity(name, score, nweights, variance_ratio=vr))
    return layers


def _make_baseline(preset: GGUFQuantType) -> dict[str, str]:
    """Build baseline map: all layers assigned to the preset type."""
    baseline: dict[str, str] = {}
    ggml_name = preset.ggml_name
    for i in range(NUM_LAYERS):
        for suffix in LAYER_WEIGHTS:
            hf_name = f"model.layers.{i}.{suffix}"
            gguf_name = _hf_to_gguf_name(hf_name)
            baseline[gguf_name] = ggml_name
    return baseline


def main() -> None:
    layers = _make_layers()
    sensitivity = SensitivityResult(
        model_id="Qwen/Qwen3-0.6B",
        metric="max_activation_variance",
        layers=layers,
    )

    print(f"Synthetic model: {NUM_LAYERS} layers, {len(layers)} tensors")
    print(f"Total weights: {sum(l.num_weights for l in layers):,}")
    print()

    for preset in PRESETS:
        print("=" * 70)
        print(f"PRESET: {preset.value} ({get_bpw(preset):.4f} bpw)")
        print("=" * 70)
        baseline = _make_baseline(preset)

        plan = two_phase_assign(
            baseline_map=baseline,
            sensitivity_result=sensitivity,
            has_imatrix=True,  # allow IQ types
            adaptive_bands=True,
        )

        # Full distribution
        dist = plan.type_distribution
        print(f"\n  Avg BPW: {plan.avg_bpw:.3f}")
        print(f"  {'Type':<12} {'Count':>5}  {'BPW':>6}  {'Pct':>5}")
        print(f"  {'-'*12} {'-'*5}  {'-'*6}  {'-'*5}")
        total = sum(dist.values())
        for qtype_name in sorted(dist.keys(), key=lambda n: get_bpw(GGUFQuantType(n))):
            count = dist[qtype_name]
            bpw = get_bpw(GGUFQuantType(qtype_name))
            pct = count / total * 100
            print(f"  {qtype_name:<12} {count:>5}  {bpw:>6.4f}  {pct:>4.0f}%")
        print()


if __name__ == "__main__":
    main()
