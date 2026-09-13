# Setup

This guide installs the current QDrive environment. Run the commands from the repository root.

The recommended path is to let `setup_env.py` detect the available hardware and select the matching PyTorch index. It
then runs `uv sync` for the base project dependencies. Optional SB3-Contrib and Trackmania/TMRL integrations are
installed separately.

## Requirements

- Python **3.13**. QDrive currently requires `==3.13.*`.
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.
- A native C++ compiler. Gymnasium builds the Box2D dependency from source on Python 3.13; SWIG is supplied by the
  project environment through `uv`.
- A GPU is recommended for image-based training, but CPU execution is supported.

## 1. Install uv

If `uv` is not already available, install it with Python or follow
the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/):

```bash
python -m pip install --upgrade uv
```

Verify that it is on `PATH`:

```bash
uv --version
```

## 2. Install the native build toolchain

QDrive depends on `gymnasium[all]`. On Python 3.13, its Box2D component may need to compile `box2d-py`, so a native C++
compiler is still required. SWIG does not need to be installed through the operating system: QDrive declares it in
`build-system.requires`, and Gymnasium's `all` extra also resolves it through `uv`.

### Windows

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools
```

In the Visual Studio Build Tools installer, select the **Desktop development with C++** workload so that `cl.exe` is
installed. A Developer PowerShell or Developer Command Prompt may be needed for `cl.exe` to be visible during a build.

Restart the terminal or IDE after installing the tools.

### Linux

Debian/Ubuntu:

```bash
sudo apt update
sudo apt install build-essential
```

Fedora:

```bash
sudo dnf install gcc-c++ make
```

## 3. Run the setup script

From the repository root:

```bash
python setup_env.py
```

The script checks for `nvidia-smi`, Linux AMD hardware, and `rocm-smi`, then updates the PyTorch source entries in
`pyproject.toml` and runs `uv sync`. That sync installs the project-local SWIG package and exposes its executable
through the `.venv` environment.

| Detected hardware                   | Platform      | Selected PyTorch build |
|-------------------------------------|---------------|------------------------|
| NVIDIA GPU detected by `nvidia-smi` | Windows/Linux | CUDA 13.2              |
| AMD GPU with ROCm and `rocm-smi`    | Linux         | ROCm 7.2               |
| AMD GPU without ROCm                | Linux         | CPU fallback           |
| AMD GPU                             | Windows       | CPU fallback           |
| No supported GPU detected           | Windows/Linux | CPU                    |

On NVIDIA systems, the script warns when the driver reports support for a CUDA version older than 13.2. Update the
NVIDIA driver if the installed PyTorch build cannot start.

On Linux, ROCm must be installed separately. If `rocm-smi` is not available, the script deliberately selects the CPU
build; install ROCm from the [AMD documentation](https://rocm.docs.amd.com/) and rerun the script when it is ready.

The setup script installs only the base dependencies. Add optional integrations after it completes:

```bash
# TRPO, TQC, RecurrentPPO, and other SB3-Contrib algorithms
uv sync --extra sb3-contrib

# Trackmania/TMRL environment integration
uv sync --extra trackmania

# Both optional integrations
uv sync --extra sb3-contrib --extra trackmania
```

The `trackmania` extra installs the Python package only. Trackmania, OpenPlanet, and the TMRL-side configuration are
separate prerequisites. QDrive registers `TrackmaniaTMRL-v0` only when TMRL is installed and available.

## 4. Verify the installation

The following check works for NVIDIA, ROCm, and CPU-only installations:

```bash
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('Accelerator available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Typical build suffixes are:

| Installation   | Version suffix | `torch.cuda.is_available()` |
|----------------|----------------|-----------------------------|
| NVIDIA         | `+cu132`       | `True`                      |
| Linux AMD/ROCm | `+rocm7.2`     | `True`                      |
| CPU            | `+cpu`         | `False`                     |

ROCm uses PyTorch's CUDA-compatible API, so `torch.cuda.is_available()` is expected to be `True` for a working ROCm
installation.

Check the core imports:

```bash
uv run python -c "import gymnasium, nicegui, stable_baselines3, wandb; print('Core imports: OK')"
```

Confirm that the project-local SWIG executable is available:

```bash
uv run swig -version
```

This should report the SWIG version installed by `uv`; a separate `swig` installation through `winget`, `apt`, or `dnf`
is not normally necessary.

Optionally smoke-test the CarRacing environment:

```bash
uv run python -c "import gymnasium as gym; env = gym.make('CarRacing-v3', render_mode='rgb_array'); observation, _info = env.reset(seed=0); print('CarRacing observation shape:', observation.shape); env.close()"
```

If the optional TMRL extra is installed, verify it separately:

```bash
uv run python -c "import tmrl; print('TMRL import: OK')"
```

## Manual setup (optional)

`setup_env.py` is preferred because it updates all relevant PyTorch source entries and handles the ROCm index visibility
needed by the resolver. For a manual setup:

1. Open `pyproject.toml`.
2. Select one PyTorch index for the `torch` and `torchvision` entries under `[tool.uv.sources]`:

    - `pytorch-cu132` for NVIDIA.
    - `pytorch-rocm` for Linux AMD with ROCm 7.2.
    - `pytorch-cpu` for CPU-only execution or Windows AMD fallback.

3. Keep the `triton-rocm` source entry consistent with the selected index when it is present. For ROCm, the
   `pytorch-rocm` index must be searchable for transitive dependencies; this is handled automatically by the setup
   script.
4. Install the base environment:

   ```bash
   uv sync
   ```

5. Add the optional extras if required:

   ```bash
   uv sync --extra sb3-contrib --extra trackmania
   ```

The three PyTorch indexes are already declared in `pyproject.toml`; normally only the source selection needs to change.

## W&B configuration

Training initializes a Weights & Biases run and logs metrics, models, and configuration artifacts. Log in with W&B or
provide settings through a local `.env` file, which is ignored by Git:

```dotenv
WANDB_API_KEY=your-key
WANDB_PROJECT=your-project
WANDB_ENTITY=your-entity
```

For an offline run, W&B's standard `WANDB_MODE=offline` environment setting can be used. The GUI and command-line entry
point load `.env` automatically.

## Start QDrive

After installation, launch the NiceGUI editor:

```bash
uv run python src/main.py
```

Or run a complete generated experiment configuration directly:

```bash
uv run python src/main.py --train --config_path experiments/CarRacing-v3/PPO/<experiment>.yaml
uv run python src/main.py --eval --config_path experiments/CarRacing-v3/PPO/<experiment>.yaml --mode rgb_array
```

See the [project README](../README.md) for the configuration format, output locations, and current Trackmania/TMRL
limitations.

## Troubleshooting

- **`uv` is not found:** install it and restart the terminal so the updated `PATH` is visible.
- **Box2D build errors:** confirm that `uv sync` completed, `uv run swig -version` works, and a working C++ compiler is
  installed; on Windows, use a Developer PowerShell or terminal where `cl.exe` is available.
- **CUDA initialization errors:** check the warning from `setup_env.py` and update the NVIDIA driver if it is older than
  the selected CUDA wheel.
- **TMRL import or environment errors:** install the `trackmania` extra and configure the external
  Trackmania/OpenPlanet/TMRL components; the Python dependency alone is not sufficient.
