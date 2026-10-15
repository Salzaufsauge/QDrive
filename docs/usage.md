# Usage

QDrive has three commands:

```bash
uv run qdrive [gui] [--ip IP] [--port PORT] [--offline]

uv run qdrive train CONFIG [--offline]

uv run qdrive evaluate CONFIG [--render-mode MODE]
```

Without a command, the GUI will be launched.

## GUI

```bash
uv run qdrive gui --ip 0.0.0.0 --port 9000
```

The GUI has no authentication and can start training runs. Only bind it to a public address on a trusted network.

### Train tab

1. **Pick a configuration:** Select an existing configuration and press *Load Config*. *Start as new Run* loads a saved
   config but resets progress and model path. Continue with step 5. Alternatively, leave config empty and continue with
   step 2.
2. **Environment Parameters:** Choose the desired environment. Select how many envs you want and how you want them
   vectorized.
3. **Wrapper Parameters:** Enable wrappers and set their parameters. Drag and drop wrappers to reorder them and change
   how they are applied for training/evaluation.
4. **Model Parameters:** Algorithm, policy, milestones, total timesteps, evaluation settings and the algorithm's
   constructor arguments.
5. **Train** starts the run, **Stop** ends it after the current step and writes the last checkpoint.

*Training Output* shows the console log and a live plot of episode rewards per environment.

### Eval tab

Select a configuration and press *Load Config*, then *Evaluate*. The policy runs deterministically and the rendered
frames are streamed into the page until *Stop* is pressed.

## CLI

```bash
uv run qdrive train configs/examples/example.yaml --offline
uv run qdrive evaluate configs/examples/example.yaml
```

To stop either train or evaluate, send SIGINT for a graceful shutdown.

- `train` trains or resumes the experiment in `CONFIG`. The path to the configuration can be relative or absolute.
- `evaluate` runs the saved model until interrupted. `--render-mode` is passed to the environment (default: `rgb_array`)

> Train will never rewrite the file you pass in. Instead, it will create a new copy in
> `experiments/<env>/<algoritm/<model>.yaml`.

## Outputs

| Path                                               | Content                                             |
|----------------------------------------------------|-----------------------------------------------------|
| `models/.../<name>.zip`                            | Best model so far (by evaluation reward)            |
| `models/.../<name>.pkl`                            | `VecNormalize` statistics for the best model        |
| `models/.../<name>_last.zip`, `<name>_last.pkl`    | Latest checkpoint and statistics, used for resuming |
| `models/.../<name>_replay_buffer.pkl`              | Replay buffer (off-policy algorithms, if enabled)   |
| `models/.../<name>/<milestone>/`                   | Model saved at each milestone                       |
| `experiments/.../<name>.yaml`                      | Experiment state and configuration                  |
| `experiments/.../<name>/<milestone>/`, `.../best/` | Evaluation videos                                   |
| `wandb/`                                           | Weights & Biases run data                           |

