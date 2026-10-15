# Setup

## Requirements

- Python **3**: To run the setup script
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- A C++ compiler: `gymnasium[all]` builds `box2d-py` from source.

## Install

```bash
python setup_env.py
```

The script picks the PyTorch build for your machine and runs uv sync:

| Detected                   | Build     |
|----------------------------|-----------|
| `nvidia-smi` found         | CUDA 13.2 |
| Linux and `rocm-smi` found | ROCM 7.2  |
| anything else              | CPU       |

> In its current form, the script changes the tool.uv.sources for the torch packages. This is problematic but works.

### Optional integrations

| Extra         | Adds                                                                                                                                         |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `sb3-contrib` | Adds support for experimental algorithms and wrappers from [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) |
| `trackmania`  | TMRL (see [Trackmania](trackmania.md))                                                                                                       |

```bash
uv sync --extra sb3-contrib
```

### Development tools

| Group | Adds                                          |
|-------|-----------------------------------------------|
| dev   | Adds general development tools such as pytest |
| lint  | Adds tools for formatting such as ruff        |

```bash
uv sync --group lint
```

## Verify

```bash
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('Accelerator available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`torch.cuda.is_available()` is True for both CUDA and ROCm

## Weights & Biases

Training logs to [Weights & Biases](https://wandb.ai/site). Either run `wandb login`, or create a git-ignored `.env`:

```dotenv
WANDB_API_KEY=your-key
WANDB_PROJECT=your-project
WANDB_ENTITY=your-entity
```

Add `--offline` when training over cli or the gui to skip the network entirely. The runs are stored under wandb and can
later be uploaded using `wandb sync`

## Troubleshooting

- **`uv` not found:** install it and restart the terminal
- **`box2d-py` fails to build:** check that a C++ compiler is available and that `uv run swig -version` works
- **Trackmania import errors:** see [Trackmania](trackmania.md) as the Python package alone is not enough to run the
  environment