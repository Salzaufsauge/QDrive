import ast
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
    tree = ast.parse(str(s), mode="eval")

    if isinstance(tree.body, ast.Lambda):
        validate_ast(tree)
        return eval(compile(tree, "<lambda>", "eval"))

    return s


def parse_params(data):
    if isinstance(data, dict):
        return {k: parse_params(v) for k, v in data.items()}
    return parse_lambda(data)
