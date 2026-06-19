# Setup

This guide explains how to set up QDrive on your machine.

**Recommended:** Follow steps 1–4. The setup script detects your GPU automatically and installs the correct PyTorch version.

**Alternative:** If you prefer not to use the script, skip step 3 and go to [Manual Setup](#manual-setup-optional) instead. Steps 1, 2 and 4 still apply.

## Table of Contents

- [1. Install uv](#1-install-uv)
- [2. Install Build Tools (SWIG & C++ Compiler)](#2-install-build-tools-swig--c-compiler)
- [3. Run the Setup Script](#3-run-the-setup-script)
- [4. Verify Installation](#4-verify-installation)
- [Manual Setup (Optional)](#manual-setup-optional)

---

## Requirements

* Python 3.13
* [uv](https://docs.astral.sh/uv/) (package manager)
* [SWIG](https://www.swig.org/) and a C++ compiler (required to build Box2D for `gymnasium[box2d]`)
* A GPU is recommended but not required

---

## 1. Install uv

If you do not have `uv` installed, install it via pip (e.g. in the PyCharm terminal):
```
pip install uv
```

## 2. Install Build Tools (SWIG & C++ Compiler)

There is no prebuilt `box2d-py` wheel for Python 3.13, so `gymnasium[box2d]` builds **Box2D** from
source. This needs **two** tools: **SWIG** to generate the Python bindings and a **C++ compiler** to
build the extension.

**Windows:**
```
winget install swig                                    # SWIG
winget install Microsoft.VisualStudio.2022.BuildTools  # C++ compiler
```
> In the Visual Studio Build Tools installer, select the **"Desktop development with C++"** workload,
> otherwise `cl.exe` is not installed. Alternatively, get SWIG as a binary from
> [swig.org](https://www.swig.org/download.html) and add it to your `PATH`.
>
> Restart your IDE and terminal afterward so the updated `PATH` is picked up.

**Linux:**
```
sudo apt install swig build-essential   # Debian/Ubuntu
sudo dnf install swig gcc-c++ make      # Fedora
```

Verify SWIG is available:
```
swig -version
```
---

## 3. Run the Setup Script

The setup script automatically detects your GPU and installs the correct PyTorch version. It is a
plain Python script and runs the same way on Windows and Linux:

```
python setup_env.py
```

| Detected hardware | OS            | PyTorch build installed |
|-------------------|---------------|-------------------------|
| NVIDIA GPU        | Windows/Linux | CUDA 13.2               |
| AMD GPU           | Windows       | CPU only ¹              |
| AMD GPU + ROCm    | Linux         | ROCm 7.2                |
| AMD GPU, no ROCm  | Linux         | CPU only ²              |
| No GPU            | any           | CPU only                |

> ¹ ROCm is only supported on Linux. On Windows with an AMD GPU, the CPU build is used as fallback.
> For GPU acceleration on Windows with AMD, install `torch-directml` manually after setup.
>
> ² On Linux, ROCm must be installed separately (the script checks for `rocm-smi`). If an AMD GPU is
> detected but ROCm is missing, the CPU build is used. Install ROCm from https://rocm.docs.amd.com
> and re-run the script to get the ROCm build.

The script then runs `uv sync` to install all remaining dependencies.

---

## 4. Verify Installation

**NVIDIA & AMD:**

```
uv run python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output NVIDIA:
```
2.x.x+cu132
CUDA available: True
GPU: NVIDIA GeForce ...
```

Expected output AMD (Linux + ROCm):
```
2.x.x+rocm7.2
CUDA available: True
GPU: AMD Radeon ...
```

> ROCm uses PyTorch's CUDA API, so `torch.cuda.is_available()` reports `True` and the version
> string ends in `+rocm7.2`. On a CPU-only build the version ends in `+cpu` and
> `CUDA available:` is `False`.
---

## Manual Setup (optional)

If you prefer to install manually, change index to only one of the following `pyproject.toml`:

```
[tool.uv.sources]
torch = [{ index = "pytorch-cu132" }]   # NVIDIA
torch = [{ index = "pytorch-rocm" }]    # AMD
torch = [{ index = "pytorch-cpu" }]     # CPU only
```

Then run:
```
uv sync
```
---
