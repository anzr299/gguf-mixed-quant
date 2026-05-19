"""Parse llama-quantize output to extract baseline per-tensor type assignments."""

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TensorAssignment:
    """A single tensor's quantization assignment from llama-quantize."""

    tensor_name: str
    ggml_type: str
    original_type: str
    size_mib_before: float
    size_mib_after: float


def parse_quantize_output(output: str) -> list[TensorAssignment]:
    """
    Parse llama-quantize stdout to extract per-tensor type assignments.

    Handles two formats:
      Regular: [3/147] name - [...], type = f16, converting to q6_K .. size = 501.00 MiB -> 205.49 MiB
      Dry-run: [3/147] name - [...], type = bf16, size = 970.00 MiB -> 397.85 MiB (q6_K)

    :param output: Full stdout from llama-quantize.
    :return: List of TensorAssignment for quantized tensors.
    """
    # Match lines with "converting to" (regular quantization)
    pattern_regular = re.compile(
        r"\[\s*\d+/\s*\d+\]\s+"
        r"(\S+)"                                  # tensor name
        r"\s+-\s+\[.*?\],\s+type\s+=\s+(\w+),\s+"
        r"converting to\s+(\w+)"                  # target ggml type
        r"\s+\.\.\s+size\s+=\s+([\d.]+)\s+MiB"    # size before
        r"\s+->\s+([\d.]+)\s+MiB"                  # size after
    )

    # Match dry-run format: size = X MiB -> Y MiB (type)
    pattern_dryrun = re.compile(
        r"\[\s*\d+/\s*\d+\]\s+"
        r"(\S+)"                                  # tensor name
        r"\s+-\s+\[.*?\],\s+type\s+=\s+(\w+),\s+"
        r"size\s+=\s+([\d.]+)\s+MiB"              # size before
        r"\s+->\s+([\d.]+)\s+MiB"                  # size after
        r"\s+\((\w+)\)"                           # target ggml type
    )

    assignments = []
    for match in pattern_regular.finditer(output):
        assignments.append(TensorAssignment(
            tensor_name=match.group(1),
            original_type=match.group(2),
            ggml_type=match.group(3),
            size_mib_before=float(match.group(4)),
            size_mib_after=float(match.group(5)),
        ))

    if not assignments:
        for match in pattern_dryrun.finditer(output):
            assignments.append(TensorAssignment(
                tensor_name=match.group(1),
                original_type=match.group(2),
                ggml_type=match.group(5),
                size_mib_before=float(match.group(3)),
                size_mib_after=float(match.group(4)),
            ))

    return assignments


def get_baseline_assignments(
    f16_gguf: Path,
    quant_type: str,
    llama_cpp: Path,
    imatrix_path: Path | None = None,
) -> list[TensorAssignment]:
    """
    Run llama-quantize and parse its output to get the baseline tensor assignments.

    Uses --dry-run when available (fast, no disk I/O), falls back to full
    quantize-and-discard if --dry-run isn't supported.

    :param f16_gguf: Path to the F16 GGUF file.
    :param quant_type: Quantization preset name (e.g. "Q4_K").
    :param llama_cpp: Path to llama.cpp directory.
    :param imatrix_path: Optional path to an importance matrix file.
    :return: List of TensorAssignment from llama.cpp's rules.
    """
    quantize_bin = llama_cpp / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        raise FileNotFoundError(f"llama-quantize not found at {quantize_bin}")
    if not f16_gguf.exists():
        raise FileNotFoundError(f"F16 GGUF not found at {f16_gguf}")

    # Try --dry-run first (instant, no temp file needed)
    cmd = [str(quantize_bin), "--dry-run"]
    if imatrix_path is not None:
        imat = Path(imatrix_path)
        if not imat.exists():
            raise FileNotFoundError(f"Importance matrix not found at {imat}")
        cmd.extend(["--imatrix", str(imat)])
    cmd.extend([str(f16_gguf), quant_type])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        combined_output = result.stdout + result.stderr
        assignments = parse_quantize_output(combined_output)
        if assignments:
            return assignments

    # Fallback: full quantization to a temp file (for older llama.cpp builds)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=True) as tmp:
        tmp_path = tmp.name

    cmd = [str(quantize_bin)]
    if imatrix_path is not None:
        cmd.extend(["--imatrix", str(imatrix_path)])
    cmd.extend([str(f16_gguf), tmp_path, quant_type])

    result = subprocess.run(cmd, capture_output=True, text=True)

    tmp_file = Path(tmp_path)
    if tmp_file.exists():
        tmp_file.unlink()

    if result.returncode != 0:
        raise RuntimeError(
            f"llama-quantize failed (exit {result.returncode}):\n{result.stderr}"
        )

    combined_output = result.stdout + result.stderr
    assignments = parse_quantize_output(combined_output)

    if not assignments:
        raise RuntimeError(
            "Failed to parse any tensor assignments from llama-quantize output."
        )

    return assignments


def baseline_to_map(assignments: list[TensorAssignment]) -> dict[str, str]:
    """
    Convert a list of TensorAssignment to a {tensor_name: ggml_type} dict.

    :param assignments: Output from get_baseline_assignments or parse_quantize_output.
    :return: Mapping from GGUF tensor name to ggml type string.
    """
    return {a.tensor_name: a.ggml_type for a in assignments}
