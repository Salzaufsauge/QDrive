import collections.abc
import importlib
import inspect
import numbers
import pkgutil
import types
import typing
from functools import cache
from itertools import chain
from typing import get_origin

import gymnasium
import sb3_contrib
import stable_baselines3 as sb3
from nicegui import ui
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnvWrapper


def iter_modules(package):
    for _, modname, _ in pkgutil.walk_packages(
            package.__path__, package.__name__ + "."
    ):
        yield modname


def make_ui_for_param(param, value=None, visible=True):
    ann = param.annotation
    args = typing.get_args(ann)

    val = param.default if value is None else value
    val = None if val is inspect.Parameter.empty else val

    if callable(val):
        val = inspect.getsource(val).strip()

    if args:
        ann = unwrap_optional(ann)

        if any(
                typing.get_origin(a) is collections.abc.Callable
                or a is typing.Callable
                for a in args
        ):
            elem = ui.textarea(
                label=param.name,
                value=str(val) if val is not None else ''
            )

            elem.set_visibility(visible)
            return elem.classes('flex-grow')

    origin = typing.get_origin(ann)

    if ann is str:
        elem = ui.input(
            label=param.name,
            value=val
        )

    elif ann is int:
        elem = ui.number(
            label=param.name,
            value=val
        )

    elif ann is float:
        elem = ui.number(
            label=param.name,
            value=val
        )

    elif ann is bool:
        elem = ui.checkbox(
            text=param.name,
            value=val
        )

    elif origin is dict:
        rows = [
            {"key": k, "value": v}
            for k, v in (val or {}).items()
        ]

        elem = ui.table(
            columns=[
                {"name": "key", "label": "key", "field": "key"},
                {"name": "value", "label": "value", "field": "value"},
            ],
            rows=rows
        )

    elif origin is typing.Callable:
        elem = ui.textarea(
            label=param.name,
            value=str(val) if val is not None else ''
        )

    elif isinstance(val, bool):
        elem = ui.checkbox(
            text=param.name,
            value=val
        )

    elif isinstance(val, numbers.Number):
        elem = ui.number(
            label=param.name,
            value=val
        )

    elif param.name.endswith("keys"):
        rows = [
            {"key": k, "value": v}
            for k, v in (val or {}).items()
        ]

        elem = ui.table(
            columns=[
                {"name": "key", "label": "key", "field": "key"},
                {"name": "value", "label": "value", "field": "value"},
            ],
            rows=rows
        )

    else:
        elem = ui.input(
            label=f"{param.name} (unknown type)",
            value=str(val) if val is not None else ''
        )

    elem.set_visibility(visible)
    return elem.classes('flex-grow')


@cache
def load_algorithms():
    algos = {}

    for modname in chain(iter_modules(sb3), iter_modules(sb3_contrib)):
        try:
            module = importlib.import_module(modname)
        except Exception:
            continue

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                    issubclass(obj, BaseAlgorithm)
                    and obj is not BaseAlgorithm
                    and obj is not OffPolicyAlgorithm
                    and obj is not OnPolicyAlgorithm
            ):
                algos[name] = obj

    return algos


@cache
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


@cache
def load_env_wrappers():
    wrappers = {}

    for modname in chain(
            iter_modules(sb3.common),
            iter_modules(sb3_contrib),
            iter_modules(gymnasium.wrappers)
    ):
        try:
            module = importlib.import_module(modname)
        except Exception:
            continue

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                    issubclass(obj, gymnasium.Wrapper)
                    and obj is not gymnasium.Wrapper
            ) or (
                    issubclass(obj, VecEnvWrapper)
                    and obj is not VecEnvWrapper
            ):
                wrappers[name] = obj

    return wrappers
