# Configuration

## Experiment file

A YAML file defines an experiment. It can be written either by hand or using the GUI. During training the file is
updated with progress information that can be used to resume training.

```yaml
env_param:
  env_id: CartPole-v1
  n_envs: 1
  vec_env_cls: DummyVecEnv
env_wrappers: { }
model_param:
  policy: MlpPolicy
  env: null
  learning_rate: 0.0003

algorithm: PPO
milestones:
  - '3000'
total_timesteps: 10000
save_replay_buffer: true
callback_params:
  eval_freq: 5000
  n_eval_episodes: 10
  deterministic: true
model_path: models/CartPole-v1/PPO/example.zip

```

The experiment file follows the signatures defined by Stable-Baselines3 and gymnasium. Only a few fields are required,
including the top level fields, fields not defined by signatures and fields that define the basics such as the `policy`
field.
For an example, see the [example experiment file](../configs/examples/example.yaml).

Some signatures include parameters that can accept functions/callables. For example, the `learning_rate` field. In these
cases it is possible to pass a lambda in the form of `lambda x: x`.
For details on which parameters support functions
see [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/).

## Discovery

The discovery config contains the packages in which QDrive should search for wrappers and algorithms.
> Only algorithms that subclass `BaseAlgorithm` become selectable algorithms.

## Overrides

Forces wrapper arguments depending on the mode, so a model is evaluated the way it should be.