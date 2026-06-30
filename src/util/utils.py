from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def get_envs():
    return sorted([env_id for env_id in gym.envs.registry.keys()])


def get_project_root():
    return Path(__file__).parent.parent.parent


def get_vec_env_class(cls):
    match cls:
        case "DummyVecEnv":
            return DummyVecEnv
        case "SubprocVecEnv":
            return SubprocVecEnv
    return None


def replace_empty_strings(obj):
    if isinstance(obj, dict):
        return {k: replace_empty_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_empty_strings(item) for item in obj]
    elif obj == "":
        return None
    else:
        return obj
