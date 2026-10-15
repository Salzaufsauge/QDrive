# Trackmania (TMRL)

QDrive can train on Trackmania 2020 using the real-time gymnasium environment provided
by [TMRL](https://github.com/trackmania-rl/tmrl).

## Requirements

TMRL needs Trackmania 2020, [OpenPlanet](https://openplanet.dev/) and TMRL's own configuration files. For detailed
instructions on how to set this up,
follow the [TMRL installation guide](https://github.com/trackmania-rl/tmrl/blob/master/readme/Install.md) first,
excluding the installation of TMRL itself.

This will be done for you by running:

```bash
# install tmrl extra dependency
uv sync --extra trackmania
# setup TMRLData
uv run python -m tmrl --install
```

The Trackmania gymnasium environment `TrackmaniaTMRL-v0` is registered and appears in the environment list only if
`tmrl` itself is importable.

## Integration

- **Observation adapter**: `TMRLFullObsWrapper` converts the raw TMRL observation into a dictionary of `speed`, `gear`,
  `rpm`, `image_history` and `action_history` so MultiInputPolicy can be used. With `use_only_images` it returns only
  the image stack which allows the usage of CNNPolicies.
- **Single environment**: The game is real-time and exclusive, thus QDrive forces a single `DummyVecEnv` environment.
  This environment is reused during evaluation.

## Limitations

- While TMRL provides a way to separate trainer from worker, it does not expose this feature to the rt-gymnasium
  environment. Thus training in QDrive can be slower and less efficient than in TMRL.
- Due to how the environment is handled, it might be necessary to restart the TMRL_GrabData plugin as otherwise it fails
  to reconnect on startup.