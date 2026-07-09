import copy
from enum import Enum

import gymnasium
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, VecCheckNan, VecEnvWrapper

from backend.config.config import ExperimentConfig
from util.inspection_helper import load_env_wrappers
from util.utils import get_vec_env_class


class EnvMode(Enum):
    TRAIN = "train"
    EVAL = "eval"


def build_env(config: ExperimentConfig, mode: EnvMode) -> VecEnv:
    env_config = copy.deepcopy(config.env_params)
    wrapper_config = copy.deepcopy(config.env_wrappers)

    env_override = {
        "vec_env_cls": get_vec_env_class(env_config["vec_env_cls"]) if mode == EnvMode.TRAIN else DummyVecEnv,
        "n_envs": env_config["n_envs"] if mode == EnvMode.TRAIN else 1,
    }
    env = make_vec_env(**(env_config | env_override))

    env = build_wrapper(wrapper_config, mode, env)

    if mode == EnvMode.TRAIN:
        env = VecCheckNan(env)

    return env


def build_wrapper(wrapper_config: list, mode: EnvMode, env) -> VecEnv:
    wrappers = load_env_wrappers()

    for wrapper in wrapper_config:
        for wrapper_name, params in wrapper.items():
            env = wrap(env, wrappers[wrapper_name], **params)

    return env


def wrap(env, wrapper_cls, **kwargs):
    if issubclass(wrapper_cls, gymnasium.Wrapper):
        return wrapper_cls(env=env, **kwargs)
    elif issubclass(wrapper_cls, VecEnvWrapper):
        return wrapper_cls(venv=env, **kwargs)

    raise TypeError("Unknown wrapper type")
