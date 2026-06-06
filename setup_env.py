"""Detect the GPU and set the matching PyTorch index in pyproject.toml.

NVIDIA (Windows/Linux)-> CUDA
Linux + AMD + ROCm-> ROCm
Linux + AMD without ROCm-> CPU (with hint)
Otherwise (incl. Windows + AMD, no GPU)-> CPU
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

TOML_PATH = Path(__file__).parent / "pyproject.toml"


def set_torch_index(index_name: str) -> None:
    content = TOML_PATH.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'(index\s*=\s*")pytorch-\w+(")', rf"\g<1>{index_name}\g<2>", content
    )
    if count == 0:
        sys.exit(f"ERROR: no PyTorch index entry found in {TOML_PATH}")
    TOML_PATH.write_text(updated, encoding="utf-8")
    print(f"PyTorch index set to: {index_name} ({count} entries updated)")


def set_rocm_index_explicit(explicit: bool) -> None:
    # ROCm builds require triton-rocm as a transitive dependency, which only
    # exists in the pytorch-rocm index. Removing explicit = true lets uv search
    # that index for all packages (including transitive ones). For non-ROCm
    # builds the flag is restored so the index is not searched unnecessarily.
    content = TOML_PATH.read_text(encoding="utf-8")

    def rewrite_block(m: re.Match) -> str:
        block = m.group(0)
        block = re.sub(r'\nexplicit\s*=\s*\S+', '', block)
        if explicit:
            block = block.rstrip('\n') + '\nexplicit = true\n'
        return block

    updated = re.sub(
        r'\[\[tool\.uv\.index\]\][ \t]*\nname\s*=\s*"pytorch-rocm"[^\[]*',
        rewrite_block,
        content,
        flags=re.DOTALL,
    )
    TOML_PATH.write_text(updated, encoding="utf-8")


def amd_on_linux() -> bool:
    try:
        out = subprocess.run(["lspci"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return False
    return bool(re.search(r"AMD|Radeon|Instinct", out, re.IGNORECASE))


def setup() -> None:
    is_linux = sys.platform.startswith("linux")

    # NVIDIA (Windows/Linux) -> CUDA
    if shutil.which("nvidia-smi"):
        set_torch_index("pytorch-cu132")
        set_rocm_index_explicit(True)
        print("Detected: NVIDIA GPU")

    # Linux + AMD with ROCm -> ROCm
    elif is_linux and shutil.which("rocm-smi") and amd_on_linux():
        set_torch_index("pytorch-rocm")
        set_rocm_index_explicit(False)
        print("Detected: AMD GPU (Linux), using ROCm")

    # Linux + AMD without ROCm -> CPU (with hint)
    elif is_linux and amd_on_linux():
        print("Detected: AMD GPU (Linux), ROCm not installed, using CPU build")
        print("HINT: Install ROCm for GPU support: https://rocm.docs.amd.com")
        print("      Then re-run this script for ROCm support.")
        set_torch_index("pytorch-cpu")
        set_rocm_index_explicit(True)

    # Otherwise (Windows + AMD, no GPU) -> CPU
    else:
        set_torch_index("pytorch-cpu")
        set_rocm_index_explicit(True)
        print("No NVIDIA/ROCm setup detected, using CPU build")

    if shutil.which("uv") is None:
        sys.exit(
            "ERROR: 'uv' was not found on your PATH.\n"
            "The index in pyproject.toml was set, but 'uv sync' cannot run.\n"
            "Install uv (https://docs.astral.sh/uv/) or add it to your PATH, then run 'uv sync'."
        )

    subprocess.run(["uv", "sync"], check=True)


if __name__ == "__main__":
    setup()
