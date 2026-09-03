import copy
import datetime
import re
import sys
from functools import cache
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import yaml
from nicegui import ui
from stable_baselines3.common import noise
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
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


def build_action_noise(spec: dict, env):
    spec = dict(spec or {})
    noise_type = spec.pop("type", None)

    if noise_type is None:
        return None

    if noise_type not in ("NormalActionNoise", "OrnsteinUhlenbeckActionNoise"):
        raise ValueError(f"Unknown noise type: {noise_type}")

    n_actions = env.action_space.shape[-1]
    mean = float(spec.pop("mean", 0.0))
    sigma = float(spec.pop("sigma", 0.1))

    return getattr(noise, noise_type)(
        mean=mean * np.ones(n_actions),
        sigma=sigma * np.ones(n_actions),
        **spec,
    )

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
    src = model if isinstance(model, VecNormalize) else model.get_vec_normalize_env()
    dst = unwrap_vec_normalize(target_env)
    if src is None or dst is None:
        return

    if getattr(src, "obs_rms", None) is not None:
        dst.obs_rms = copy.deepcopy(src.obs_rms)

    if getattr(src, "ret_rms", None) is not None:
        dst.ret_rms = copy.deepcopy(src.ret_rms)


def load_vecnorm_stats(vecnorm_path, target_env):
    dst = unwrap_vec_normalize(target_env)
    if dst is None:
        return

    loaded = VecNormalize.load(vecnorm_path, venv=dst.venv)
    copy_vecnorm(loaded, target_env)


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


def cleanup_ansi(text: str) -> str:
    # Source - https://stackoverflow.com/a/14693789
    # Posted by Martijn Pieters, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-08-21, License - CC BY-SA 4.0

    ansi_escape = re.compile(
        r"\r?\x1B\[Am\x1B\[2K\r?"
        r"|\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
    )
    return ansi_escape.sub("", text)
