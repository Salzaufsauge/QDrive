"""GPU automatisch erkennen und passenden PyTorch-Index in pyproject.toml setzen.

NVIDIA (Windows/Linux)-> CUDA
Linux + AMD + ROCm-> ROCm
Linux + AMD ohne ROCm-> CPU (mit Hinweis)
Sonst (inkl. Windows + AMD, kein GPU)-> CPU
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

TOML_PATH = Path(__file__).parent / "pyproject.toml"

def set_torch_index(index_name: str) -> None:
    content = TOML_PATH.read_text(encoding="utf-8")
    updated = re.sub(r'(index\s*=\s*")pytorch-\w+(")', rf"\g<1>{index_name}\g<2>", content)
    TOML_PATH.write_text(updated, encoding="utf-8")
    print(f"PyTorch Index gesetzt auf: {index_name}")

def amd_on_linux() -> bool:
    try:
        out = subprocess.run(["lspci"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return False
    return bool(re.search(r"AMD|Radeon|Instinct", out, re.IGNORECASE))


def main() -> None:
    is_linux = sys.platform.startswith("linux")

    # NVIDIA (Windows/Linux) -> CUDA
    if shutil.which("nvidia-smi"):
        set_torch_index("pytorch-cu132")
        print("Erkannt: NVIDIA GPU")

    # Linux + AMD mit ROCm -> ROCm
    elif is_linux and shutil.which("rocm-smi") and amd_on_linux():
        set_torch_index("pytorch-rocm")
        print("Erkannt: AMD GPU (Linux), ROCm wird verwendet")

    # Linux + AMD ohne ROCm -> CPU (mit Hinweis)
    elif is_linux and amd_on_linux():
        print("Erkannt: AMD GPU (Linux), ROCm nicht installiert, CPU-Build wird verwendet")
        print("HINWEIS: Installiere ROCm fuer GPU-Support: https://rocm.docs.amd.com")
        print("         Danach dieses Skript erneut ausfuehren fuer ROCm-Unterstuetzung.")
        set_torch_index("pytorch-cpu")

    # Sonst (Windows + AMD, kein GPU) -> CPU
    else:
        set_torch_index("pytorch-cpu")
        print("Kein NVIDIA/ROCm-Setup erkannt, CPU-Build wird verwendet")

    subprocess.run(["uv", "sync"], check=True)

if __name__ == "__main__":
    main()