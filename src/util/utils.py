import ast
import copy
import sys
from functools import cache
from pathlib import Path

import cv2
import gymnasium as gym
import yaml
from nicegui import ui
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, unwrap_vec_normalize

from util.inspection_helper import resolve_name


@cache
def get_envs():
    return sorted([env_id for env_id in gym.envs.registry.keys()])


@cache
def get_project_root():
    return Path(__file__).parent.parent.parent


@cache
def load_overrides():
    return yaml.safe_load((get_project_root() / "configs/overrides.yaml").read_text())

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


ALLOWED_NODES = {
    ast.Expression,
    ast.Lambda,
    ast.arguments,
    ast.arg,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Name,
    ast.Load,
    ast.Constant,
}


def validate_ast(node):
    for child in ast.walk(node):
        if type(child) not in ALLOWED_NODES:
            raise ValueError(f"Unsupported expression: {type(child).__name__}")


def get_config_path(model_path):
    model_path = Path(model_path)
    parts = list(model_path.parts)

    if "models" in parts:
        i = parts.index("models")
        parts[i] = "experiments"

    conf_path = Path(*parts)
    conf_path = conf_path.with_suffix(".yaml")

    return conf_path


def parse_val(s: str):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


# useful for stuff like lr where a lambda can be passed
def parse_lambda(s):
    if not isinstance(s, str):
        return s

    tree = ast.parse(s, mode="eval")

    if isinstance(tree.body, ast.Lambda):
        validate_ast(tree)
        return eval(compile(tree, "<lambda>", "eval"), {"__builtins__": {}})

    try:
        if isinstance(tree.body, (ast.Name, ast.Attribute)):
            return resolve_name(s)
    except Exception as e:
        print(e, file=sys.stderr)
        print(f"Using {s} as is")

    return s


def parse_params(data):
    if isinstance(data, dict):
        return {k: parse_params(v) for k, v in data.items()}
    return parse_lambda(data)


def build_ui_params(params: list, elem_per_row: int, action):
    temp = []

    for i in range(0, len(params), elem_per_row):
        with ui.row().classes("w-full"):
            for param in params[i:i + elem_per_row]:
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
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()
