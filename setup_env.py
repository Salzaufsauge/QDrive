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


CUDA_INDEX_VERSION = (13, 2)

def set_torch_index(index_name: str) -> None:
    content = TOML_PATH.read_text(encoding="utf-8", newline="")
    updated, count = re.subn(r'(index\s*=\s*")pytorch-\w+(")', rf"\g<1>{index_name}\g<2>", content)
    if count == 0:
        sys.exit(f"ERROR: no PyTorch index entry found in {TOML_PATH}")
    TOML_PATH.write_text(updated, encoding="utf-8", newline="")
    print(f"PyTorch index set to: {index_name} ({count} entries updated)")

def amd_on_linux() -> bool:
    try:
        out = subprocess.run(["lspci"], capture_output=True, text=True, check=False).stdout
    except FileNotFoundError:
        return False
    return bool(re.search(r"AMD|Radeon|Instinct", out, re.IGNORECASE))

def nvidia_cuda_version() -> tuple[int, int] | None:
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False).stdout
    except FileNotFoundError:
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))

def setup() -> None:
    if shutil.which("uv") is None:
        sys.exit(
            "ERROR: 'uv' was not found on your PATH.\n"
            "Install uv (https://docs.astral.sh/uv/) or add it to your PATH, then re-run."
        )
    is_linux = sys.platform.startswith("linux")

    # NVIDIA (Windows/Linux) -> CUDA
    if shutil.which("nvidia-smi"):
        print("Detected: NVIDIA GPU")
        driver_cuda = nvidia_cuda_version()
        if driver_cuda is not None and driver_cuda < CUDA_INDEX_VERSION:
            print(
                f"WARNING: driver supports up to CUDA {driver_cuda[0]}.{driver_cuda[1]}, "
                f"but the wheels target CUDA {CUDA_INDEX_VERSION[0]}.{CUDA_INDEX_VERSION[1]}.\n"
                "         torch may fail to run; update your NVIDIA driver if so."
            )
        set_torch_index("pytorch-cu132")

    # Linux + AMD with ROCm -> ROCm
    elif is_linux and shutil.which("rocm-smi") and amd_on_linux():
        print("Detected: AMD GPU (Linux), using ROCm")
        set_torch_index("pytorch-rocm")

    # Linux + AMD without ROCm -> CPU (with hint)
    elif is_linux and amd_on_linux():
        print("Detected: AMD GPU (Linux), ROCm not installed, using CPU build")
        print("HINT: Install ROCm for GPU support: https://rocm.docs.amd.com")
        print("      Then re-run this script for ROCm support.")
        set_torch_index("pytorch-cpu")

    # Otherwise (Windows + AMD, no GPU) -> CPU
    else:
        print("No NVIDIA/ROCm setup detected, using CPU build")
        set_torch_index("pytorch-cpu")

    subprocess.run(["uv", "sync"], check=True)

if __name__ == "__main__":
    setup()
