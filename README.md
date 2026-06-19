# QDrive

QDrive is a reinforcement learning project focused on training autonomous racing agents that can progress from simple simulated driving environments to competitive real-world-style racing games.

The project begins with Gymnasium's CarRacing environment to establish core driving behaviors such as:

* Track following
* Steering control
* Throttle and brake management
* Recovery from mistakes
* Generalized driving through domain randomization

The long-term objective is to transfer these learned driving skills into increasingly realistic environments, eventually targeting Trackmania and competitive performance against human players.

---

## Vision

The roadmap for QDrive is divided into multiple stages:

### Stage 1: 2D Driving Fundamentals

Train an agent in Gymnasium's `CarRacing-v3` environment using PPO and convolutional neural network policies.

Goals:

* Learn lane and track following
* Complete laps consistently
* Generalize across randomly generated tracks
* Develop stable driving behavior

### Stage 2: Enhanced Simulation

Expand training with:

* Longer training runs
* Hyperparameter tuning
* Observation preprocessing
* Reward shaping
* Curriculum learning

Goals:

* Faster lap times
* Better cornering
* Increased robustness

### Stage 3: 3D Racing

Transition to a 3D racing environment such as Trackmania.

Potential approaches:

* Screen-capture based reinforcement learning
* Direct game-state integration (if available)
* Imitation learning from replays
* Hybrid RL + supervised learning

Goals:

* Transfer driving concepts learned in 2D
* Adapt to 3D perception
* Learn advanced racing lines

### Stage 4: Human-Level Competition

Train agents capable of:

* Racing against real players
* Maintaining competitive lap times
* Adapting to new tracks
* Potentially outperforming average or advanced human drivers

Stretch goals:

* Online racing
* Multi-agent training
* Self-play
* Human replay distillation
* World-model-based driving agents

---

## Current Implementation

The current training script:

1. Creates 8 parallel `CarRacing-v3` environments.
2. Enables domain randomization.
3. Trains a PPO agent using a CNN policy.
4. Saves trained checkpoints.
5. Runs an evaluation episode with rendering enabled.

### Training Configuration

| Parameter            | Value      |
| -------------------- | ---------- |
| Algorithm            | PPO        |
| Policy               | CnnPolicy  |
| Environments         | 8          |
| Batch Size           | 1024       |
| Rollout Steps        | 4096       |
| Domain Randomization | Enabled    |
| Action Space         | Continuous |

---

## Requirements

* Python 3.13
* CUDA 13.2 compatible GPU (recommended)
* PyTorch
* Stable-Baselines3
* Swig
* Gymnasium + Box2D

Dependencies are managed using `uv`.

### Install

See the [Setup Guide](docs/setup.md) for full instructions. The setup script detects your GPU
and installs the matching PyTorch build automatically:

```bash
python setup_env.py
```

---

## Running Training

```bash
python train.py
```

The model will be saved under:

```text
models/
└── CarRacing-v3/
    └── model-YYYY-MM-DD_HH-MM-SS.zip
```

---

## Project Structure

```text
qdrive/
├── models/
├── src/
│   └── train.py
├── pyproject.toml
└── README.md
```

---

## Future Research Directions

### Reinforcement Learning

* PPO
* SAC
* Dreamer
* IMPALA
* MuZero-style world models

### Transfer Learning

* CarRacing → Trackmania
* Simulation → Game environments
* Offline replay pretraining

### Computer Vision

* End-to-end image-based control
* Representation learning
* Contrastive visual pretraining

### Competitive Racing

* Racing line optimization
* Opponent awareness
* Drafting strategies
* Risk-aware overtaking
* Multi-agent RL

---

## Current Status

🚧 Early prototype

The project currently trains PPO agents in `CarRacing-v3` and serves as the foundation for a larger autonomous racing research platform aimed at achieving competitive Trackmania performance.
