import importlib
import inspect
import pkgutil
import types
import typing
from typing import get_origin
import gradio as gr
import stable_baselines3 as sb3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy


def iter_modules(package):
    for _, modname, _ in pkgutil.walk_packages(
            package.__path__, package.__name__ + "."
    ):
        yield modname

def make_ui_for_param(param):
    ann = unwrap_optional(param.annotation)

    origin = get_origin(ann)

    if ann is str:
        return gr.Textbox(label=param.name, value=param.default)

    if ann is int:
        return gr.Number(label=param.name, value=param.default)

    if ann is float:
        return gr.Number(label=param.name, value=param.default)

    if ann is bool:
        return gr.Checkbox(label=param.name, value=param.default)

    if origin is dict:
        return gr.JSON(label=param.name, value=param.default)

    if origin is typing.Callable:
        return gr.Textbox(label=param.name, value=param.default)

    return gr.Textbox(label=f"{param.name} (unknown type)", value=param.default)

def load_algorithms():
    algos = {}

    for modname in iter_modules(sb3):
        try:
            module = importlib.import_module(modname)
        except Exception:
            continue

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                    issubclass(obj, BaseAlgorithm)
                    and obj is not BaseAlgorithm
            ):
                algos[name] = obj

    return algos


def get_policies_from_algo(algo_cls):
    policies = {}

    if hasattr(algo_cls, "policy_aliases"):
        for name, policy_cls in algo_cls.policy_aliases.items():
            policies[name] = policy_cls

    # fallback
    sig = inspect.signature(algo_cls.__init__)
    params = sig.parameters

    if "policy" in params:
        ann = params["policy"].annotation
        if inspect.isclass(ann) and issubclass(ann, BasePolicy):
            policies[ann.__name__] = ann

    return policies


def unwrap_optional(annotation):
    origin = get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is types.UnionType and type(None) in args:
        return [a for a in args if a is not type(None)][0]

    return annotation