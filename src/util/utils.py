import copy
import datetime
import sys
from functools import cache
from pathlib import Path

import cv2
import gymnasium as gym
import yaml
from nicegui import ui
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    unwrap_vec_normalize,
)


@cache
def get_envs():
    return sorted([env_id for env_id in gym.envs.registry])


@cache
def get_project_root():
    return Path(__file__).parent.parent.parent


@cache
def load_overrides():
    return yaml.safe_load((get_project_root() / "configs/overrides.yaml").read_text())


@cache
def load_discovery():
    return yaml.safe_load((get_project_root() / "configs/discovery.yaml").read_text())


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


def get_config_path(model_path):
    model_path = Path(model_path)
    parts = list(model_path.parts)

    if "models" in parts:
        i = parts.index("models")
        parts[i] = "experiments"

    conf_path = Path(*parts)
    conf_path = conf_path.with_suffix(".yaml")

    return conf_path


def build_ui_params(params: list, elem_per_row: int, action):
    temp = []

    for i in range(0, len(params), elem_per_row):
        with ui.row().classes("w-full"):
            for param in params[i : i + elem_per_row]:
                temp.append(action(param=param))

    return temp


def copy_vecnorm(model, target_env):
    src = model.get_vec_normalize_env()
    dst = unwrap_vec_normalize(target_env)
    if src is None or dst is None:
        return

    if getattr(src, "obs_rms", None) is not None:
        dst.obs_rms = copy.deepcopy(src.obs_rms)

    if getattr(src, "ret_rms", None) is not None:
        dst.ret_rms = copy.deepcopy(src.ret_rms)


def frame_to_data_url(frame):
    _, imencode_image = cv2.imencode(".jpg", frame)
    return imencode_image.tobytes()


def log(level: str, message: str):
    if level == "INFO":
        print(
            f"{datetime.datetime.now(tz=datetime.UTC).strftime('%H:%M:%S')} - {level} - {message}"
        )
    if level == "ERROR":
        print(
            f"{datetime.datetime.now(tz=datetime.UTC).strftime('%H:%M:%S')} - {level} - {message}",
            file=sys.stderr,
        )
