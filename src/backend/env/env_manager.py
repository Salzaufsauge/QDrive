import copy
from enum import Enum
from functools import partial

import gymnasium
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper

from backend.config.config import ExperimentConfig
from backend.env.tmrl_env import TMRL_ENV_ID
from util.inspection_helper import load_env_wrappers
from util.utils import get_vec_env_class, load_overrides, log


class EnvMode(Enum):
    TRAIN = "train"
    EVAL = "eval"


def build_env(config: ExperimentConfig, mode: EnvMode) -> VecEnv:
    log("INFO", f"Building environment in mode: {mode.value}")

    env_config = copy.deepcopy(config.env_params)
    wrapper_config = copy.deepcopy(config.env_wrappers)

    gym_wrappers, vec_env_wrappers = build_wrapper(wrapper_config, mode)

    only_single_instance_tmrl = env_config.get("env_id") == TMRL_ENV_ID

    env_override = {
        "vec_env_cls": get_vec_env_class(env_config["vec_env_cls"])
        if mode == EnvMode.TRAIN and not only_single_instance_tmrl
        else DummyVecEnv,
        "n_envs": env_config["n_envs"]
        if mode == EnvMode.TRAIN and not only_single_instance_tmrl
        else 1,
        "wrapper_class": compose_gym_wrappers(gym_wrappers) if gym_wrappers else None,
    }
    env = make_vec_env(**(env_config | env_override))

    for wrapper in vec_env_wrappers:
        env = wrapper(venv=env)

    return env


def build_wrapper(wrapper_config: dict, mode: EnvMode):
    wrappers = load_env_wrappers()

    gym_env_wrappers = []
    vec_env_wrappers = []

    overrides: dict = load_overrides()[mode.value]["env_wrappers"]

    for wrapper_name, params in wrapper_config.items():
        wrapper_cls = wrappers[wrapper_name]
        param_overrides = overrides.get(wrapper_name, {})
        if issubclass(wrapper_cls, gymnasium.Wrapper):
            gym_env_wrappers.append(partial(wrapper_cls, **(params | param_overrides)))
        elif issubclass(wrapper_cls, VecEnvWrapper):
            vec_env_wrappers.append(partial(wrapper_cls, **(params | param_overrides)))
        else:
            raise TypeError("Unknown wrapper type")

    return gym_env_wrappers, vec_env_wrappers


def compose_gym_wrappers(wrappers):
    def wrapper(env):
        for wrapper in wrappers:
            env = wrapper(env=env)
        return env

    return wrapper
