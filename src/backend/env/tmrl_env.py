import importlib.util
import os
import warnings

import gymnasium

TMRL_ENV_ID = "TrackmaniaTMRL-v0"


def make_tmrl_env():
    wandb_key = os.environ.get("WANDB_API_KEY")

    from tmrl import get_environment

    if wandb_key is None:
        os.environ.pop("WANDB_API_KEY", None)
    else:
        os.environ["WANDB_API_KEY"] = wandb_key

    warnings.filterwarnings("ignore", message="Time-step timed out")

    return get_environment()


def register_tmrl_env():
    if importlib.util.find_spec("tmrl") is None:
        return
    if TMRL_ENV_ID in gymnasium.registry:
        return

    gymnasium.register(
        id=TMRL_ENV_ID,
        entry_point="backend.env.tmrl_env:make_tmrl_env",
        disable_env_checker=True,
    )
