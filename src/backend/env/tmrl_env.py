import importlib.util
import warnings

import gymnasium

TMRL_ENV_ID = "TrackmaniaTMRL-v0"


def make_tmrl_env():
    from tmrl import get_environment

    warnings.filterwarnings("once", message="Time-step timed out")

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
