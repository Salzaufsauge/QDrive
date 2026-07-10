import copy
from enum import Enum
from functools import partial

import gymnasium
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, VecCheckNan, VecEnvWrapper

from backend.config.config import ExperimentConfig
from util.inspection_helper import load_env_wrappers
from util.utils import get_vec_env_class


class EnvMode(Enum):
    TRAIN = "train"
    EVAL = "eval"

class WrapperType(Enum):
    VENV = "venv"
    GYMENV = "env"


def build_env(config: ExperimentConfig, mode: EnvMode) -> VecEnv:
    env_config = copy.deepcopy(config.env_params)
    wrapper_config = copy.deepcopy(config.env_wrappers)

    gym_wrappers, vec_env_wrappers = build_wrapper(wrapper_config, mode)

    env_override = {
        "vec_env_cls": get_vec_env_class(env_config["vec_env_cls"]) if mode == EnvMode.TRAIN else DummyVecEnv,
        "n_envs": env_config["n_envs"] if mode == EnvMode.TRAIN else 1,
    }
    env = make_vec_env(**(env_config | env_override))

    for wrapper in vec_env_wrappers:
        env = wrapper(venv=env)

    if mode == EnvMode.TRAIN:
        env = VecCheckNan(env)

    return env


def build_wrapper(wrapper_config: list, mode: EnvMode):
    wrappers = load_env_wrappers()

    gym_env_wrappers = list()
    vec_env_wrappers = list()

    for wrapper in wrapper_config:
        for wrapper_name, params in wrapper.items():
            wrapper_type, wrapper_fn  = wrap(wrappers[wrapper_name], **params)
            if wrapper_type == WrapperType.GYMENV:
                gym_env_wrappers.append(wrapper_fn)
            else:
                vec_env_wrappers.append(wrapper_fn)

    return gym_env_wrappers, vec_env_wrappers


def wrap(wrapper_cls, **kwargs):
    if issubclass(wrapper_cls, gymnasium.Wrapper):
        return WrapperType.GYMENV, partial(wrapper_cls, **kwargs)
    elif issubclass(wrapper_cls, VecEnvWrapper):
        return WrapperType.VENV, partial(wrapper_cls, **kwargs)

    raise TypeError("Unknown wrapper type")
