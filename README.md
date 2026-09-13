# QDrive

QDrive is a configuration-driven reinforcement-learning application for autonomous racing agents. The current project
provides a complete Gymnasium `CarRacing-v3` workflow and an SB3 integration path for Trackmania 2020 through TMRL.

The project is considered complete for its current scope. Hyperparameter optimization and imitation learning are
possible future improvements, but neither is required to use the existing training and evaluation pipeline.

## What is included

- A NiceGUI application with separate Train and Eval views.
- YAML-based experiment configuration and resume support.
- Stable-Baselines3 training and evaluation, with optional SB3-Contrib algorithms.
- Dynamic discovery of algorithms and environment wrappers through `configs/discovery.yaml`.
- Vectorized Gymnasium environments for ordinary experiments.
- Checkpointing, `VecNormalize` statistics, off-policy replay buffers, milestone videos, and best-model tracking.
- Weights & Biases metrics, model uploads, and configuration artifacts.
- A TMRL adapter that converts the raw Trackmania observation tuple into an SB3-compatible dictionary observation.

The main self-contained example is `CarRacing-v3`. Trackmania requires a separately configured TMRL/Trackmania
environment and is described below.

## Repository layout

```text
QDrive/
├── src/main.py                 # GUI and command-line entry point
├── src/backend/
│   ├── train.py                # SB3/SB3-Contrib training orchestration
│   ├── evaluate.py             # Model evaluation loop
│   ├── controller.py           # Training/evaluation lifecycle
│   ├── callbacks/              # Streaming and milestone callbacks
│   ├── config/                 # Experiment configuration and persistence
│   └── env/                    # Environment, wrapper, and TMRL integration
├── src/frontend/               # NiceGUI editor and configuration tabs
├── src/util/                   # Discovery, logging, video, and helper code
├── configs/
│   ├── discovery.yaml          # Import modules for algorithms and wrappers
│   └── overrides.yaml          # Train/eval wrapper overrides
├── experiments/                # Generated YAML snapshots and milestone output
├── models/                     # Generated model checkpoints and statistics
├── docs/setup.md               # Detailed installation guide
├── setup_env.py                # Hardware-aware PyTorch setup
└── pyproject.toml              # Pinned project dependencies
```

`experiments/`, `models/`, `logs/`, and `wandb/` are runtime/generated data directories and are not intended to be
source-controlled. The repository keeps their directory placeholders where needed.

## Requirements

- Python **3.13**
- [`uv`](https://docs.astral.sh/uv/)
- A native C++ compiler for the Box2D dependency used by Gymnasium; SWIG is supplied by the project environment through
  `uv`
- A GPU is recommended for image-based training, but CPU execution is supported

See [`docs/setup.md`](docs/setup.md) for platform-specific build tools and the GPU detection details.

## Installation

From the repository root, the recommended setup is:

```bash
python setup_env.py
```

The setup script selects a compatible PyTorch build for the detected hardware and then runs `uv sync`. It supports
NVIDIA CUDA 13.2, Linux AMD ROCm 7.2, and CPU fallbacks. Manual installation is also possible by selecting the
appropriate PyTorch index as described in [`docs/setup.md`](docs/setup.md), then running:

```bash
uv sync
```

Install optional integrations only when they are needed:

```bash
# TRPO, TQC, RecurrentPPO, and other SB3-Contrib algorithms
uv sync --extra sb3-contrib

# Trackmania/TMRL environment integration
uv sync --extra trackmania

# Both optional integrations
uv sync --extra sb3-contrib --extra trackmania
```

## Running QDrive

### GUI

Launch the editor from the repository root:

```bash
uv run python src/main.py
```

The GUI provides Train and Eval tabs for selecting an environment, algorithm, policy, wrappers, model parameters,
callbacks, and milestones. Configurations created or saved by the application are stored below `experiments/`; model
paths normally point below `models/`.

### Command line

A complete experiment YAML can be run without opening the editor:

```bash
uv run python src/main.py --train --config_path experiments/CarRacing-v3/PPO/<experiment>.yaml
uv run python src/main.py --eval --config_path experiments/CarRacing-v3/PPO/<experiment>.yaml
```

Evaluation accepts an observation/render mode as well:

```bash
uv run python src/main.py --eval --config_path experiments/CarRacing-v3/PPO/<experiment>.yaml --mode rgb_array
```

The available command-line options are:

| Option               | Purpose                                                              |
|----------------------|----------------------------------------------------------------------|
| `--train`            | Train or resume the model described by the config.                   |
| `--eval`             | Load the configured model and run deterministic evaluation.          |
| `--config_path PATH` | Path to a complete experiment YAML, relative to the repository root. |
| `--mode MODE`        | Render mode used by evaluation; defaults to `rgb_array`.             |

`configs/overrides.yaml` contains mode-specific wrapper overrides and is not a standalone training configuration. Use a
generated experiment file or create one through the GUI.

## Configuration model

An experiment configuration contains the following main sections:

- `env_param`: environment ID, number of environments, vectorization settings, and environment arguments.
- `env_wrappers`: Gymnasium and vector-environment wrappers.
- `model_param`: algorithm constructor parameters, including the policy, learning rate, batch size, device, and policy
  kwargs.
- `algorithm`: the discovered SB3 or SB3-Contrib algorithm name.
- `callback_params`: evaluation frequency, number of evaluation episodes, and deterministic evaluation settings.
- `milestones`: optional checkpoint/video milestones.
- `total_timesteps`: target training duration.
- `model_path`: checkpoint location used for saving and resuming.

`configs/discovery.yaml` controls which Python modules are inspected for algorithms and wrappers. This keeps the editor
extensible without hard-coding every supported class into the frontend.

The repository has been used with, among others:

- SB3: PPO, SAC, DQN, and A2C.
- SB3-Contrib: TRPO, TQC, and RecurrentPPO when the `sb3-contrib` extra is installed.

## Trackmania/TMRL integration

QDrive can register `TrackmaniaTMRL-v0` when the optional `tmrl` package is installed. The `TMRLFullObsWrapper` adapts
TMRL's raw tuple observation into a dictionary containing image history, speed, gear, RPM, and action history, which
makes the full observation usable with policies such as `MultiInputPolicy`.

The TMRL environment is real-time and single-instance. QDrive therefore forces one environment and a `DummyVecEnv` for
this environment instead of attempting ordinary parallel vectorization. Trackmania, OpenPlanet, and the TMRL-side
configuration remain external prerequisites; installing the Python extra alone does not create a playable Trackmania
environment.

For a TMRL experiment using TQC, install both optional extras:

```bash
uv sync --extra sb3-contrib --extra trackmania
```

## Outputs and experiment tracking

Training writes or updates the configured model artifacts, including:

- SB3 model checkpoints (`.zip`).
- `VecNormalize` statistics (`.pkl`) when normalization is configured.
- Replay buffers for algorithms that support them.
- Saved YAML configuration snapshots and optional milestone videos below `experiments/`.
- W&B metrics and model/config artifacts when W&B is available and authenticated.

Existing generated files are local experiment state rather than required source files. A fresh checkout contains the
application and configuration templates, not the trained models.

## Current status and optional future work

The current implementation covers the intended workflow: configure an environment, train or resume an SB3 model, monitor
it, save artifacts, and evaluate it through the GUI or CLI.

Potential improvements, none of which are required for the current project, include:

- **Hyperparameter optimization:** add a sweep/parallel-run layer and standardized comparison of saved runs.
- **Imitation learning:** add a maintained behavioral-cloning or hybrid RL/IL path. There is currently no replay
  downloader, replay parser, imitation trainer, or `imitation` dependency in the repository; integrating that stack
  would require a separate compatibility decision for the pinned Gymnasium/SB3 versions.
- **Further TMRL experimentation:** improve evaluation and training procedures around the constraints of the exclusive
  real-time Trackmania environment.

## License

See [`LICENSE`](LICENSE).
