# QDrive

QDrive is a reinforcement learning framework for training autonomous racing agents, built around Gymnasium's CarRacing environment with a roadmap toward Trackmania integration.

It provides a config-driven training pipeline with an interactive GUI editor, multi-algorithm support (PPO, SAC, etc.), Wandb logging, and a foundation for skill transfer to increasingly realistic racing environments.

---

## Vision

QDrive aims to bridge the gap between simple simulated driving environments and competitive real-world-style racing games through staged development:

### Stage 1: 2D Driving Fundamentals

Train agents in Gymnasium's `CarRacing-v3` using configurable RL algorithms and policies.

Goals:
- Lane and track following
- Consistent lap completion
- Generalization across tracks via domain randomization
- Stable driving behavior

### Stage 2: Enhanced Simulation

Optimize training with:
- Hyperparameter tuning via config system
- Observation preprocessing and wrappers
- Reward shaping
- Curriculum learning

Goals:
- Faster lap times
- Better cornering
- Increased robustness

### Stage 3: 3D Racing

Transition to a 3D racing environment such as Trackmania using:
- Screen-capture based RL
- Direct game-state integration (if available)
- Imitation learning from replays
- Hybrid RL + supervised learning

### Stage 4: Human-Level Competition

Train agents capable of:
- Racing against real players
- Competitive lap times
- Adapting to new tracks

---

## Current Implementation

### Architecture

QDrive has evolved into a structured framework:

```
qdrive/
├── src/
│   ├── main.py              # CLI entry point (train/eval/gui)
│   ├── backend/
│   │   ├── controller.py    # Training/evaluation orchestration
│   │   ├── train.py         # Training logic
│   │   ├── evaluate.py      # Evaluation logic
│   │   ├── env/             # Environment management
│   │   ├── callbacks/       # Custom training callbacks
│   │   ├── config/          # Config loading and building
│   │   └── state/           # Runtime state management
│   └── frontend/
│       ├── editor.py        # NiceGUI-based interactive editor
│       └── components/      # UI components (training, eval, model tabs, etc.)
│           ├── training_tab.py
│           ├── eval_tab.py
│           ├── model_tab.py
│           ├── env_tab.py
│           ├── wrapper_tab.py
│           └── config_loader.py
├── configs/                  # Training configuration templates
│   ├── discovery.yaml        # Algorithm/env wrapper discovery config
│   └── overrides.yaml        # Override config for custom training
├── docs/                     # Documentation
│   └── setup.md
├── experiments/              # Generated experiment configs
├── models/                   # Trained model checkpoints
├── logs/                     # Training logs
├── pyproject.toml            # Project dependencies
├── setup_env.py              # Auto-setup script
└── README.md
```

### Key Features

- **Config-driven training**: YAML configs specify algorithm, policy, env wrappers, hyperparameters, etc.
- **Interactive GUI**: NiceGUI-based editor for building configs, launching training, monitoring runs, and evaluating models
- **Multi-algorithm support**: PPO, SAC, and others via Stable-Baselines3 and SB3-Contrib integration
- **Wandb logging**: Integrated Weights & Biases logging for experiments
- **Custom callbacks**: Milestone callbacks, streaming callbacks, and more
- **RL Zoo3 integration**: Access to pre-built experiment configs

### Running the Project

There are two ways to interact with QDrive:

#### Interactive GUI

Run the editor to configure and launch training visually:

```bash
python src/main.py
```

The editor provides tabs for:
- **Training**: Configure algorithms, policies, environments, and hyperparameters
- **Evaluation**: Load and evaluate trained models

#### CLI

Train directly from a config:

```bash
python src/main.py --train --config_path configs/overrides.yaml
```

Evaluate a trained model:

```bash
python src/main.py --eval --config_path experiments/<config_name>.yaml
```

Options:
- `--train` — Start training
- `--eval` — Evaluate a trained model
- `--config_path` — Path to the YAML config file
- `--mode` — Observation mode (`rgb_array`, etc.)

### Supported Algorithms and Wrappers

The project supports discovery and configuration of wrappers from:

| Source               | Algorithms | Wrappers     |
| -------------------- | ---------- | ------------ |
| Stable-Baselines3   | PPO, SAC, A2C, etc. | Yes |
| SB3-Contrib          | TRPO, TQC, etc.      | Yes |
| Gymnasium            | —                  | Yes |
| RL Zoo3              | Curated configs       | Yes |

---

## Requirements

* Python 3.13
* [uv](https://docs.astral.sh/uv/) package manager
* CUDA 13.2 GPU (recommended) or AMD GPU with ROCm 7.2 (Linux)

### Install

See the [Setup Guide](docs/setup.md) for full instructions. The setup script detects your GPU and installs the matching PyTorch build automatically:

```bash
python setup_env.py
```

### Dependencies

Managed via `uv`:

```bash
uv sync
```

Key libraries:

| Library              | Purpose                  |
| -------------------- | ------------------------ |
| Stable-Baselines3   | RL algorithms            |
| Torch + Torchvision  | GPU-accelerated training |
| Gymnasium + Box2D    | CarRacing environment    |
| SB3-Contrib          | Additional algorithms    |
| RL Zoo3              | Experiment configs       |
| Wandb                | Experiment logging       |
| NiceGUI              | Interactive editor       |
| Polars + PyArrow     | Data processing          |
| Plotly               | Visualization            |
| Ruff                 | Linting                  |

---

## Project Status

🚧 Early prototype

QDrive provides a working training pipeline for CarRacing with config-driven training and an interactive editor. The foundation is in place for scaling to more complex environments and pursuing Trackmania integration.
