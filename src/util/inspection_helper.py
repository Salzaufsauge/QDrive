import ast
import importlib
import inspect
import pkgutil
import sys
import types
import typing
from functools import cache
from typing import get_origin

import gymnasium
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnvWrapper

from util.utils import load_discovery

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


def iter_modules(package):
    if not hasattr(package, "__path__"):
        yield package.__name__
        return
    for _, modname, _ in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        yield modname


def discover_classes(packages, predicate):
    """
    Discovers classes in given packages that satisfy the provided predicate.

    :param packages: source packages to search
    :param predicate: filter to apply
    :return: dict of discovered classes
    """
    found = {}
    saved = (
        sys.argv
    )  # fix because rl_zoo3 and others parse sys.argv on import which causes issues
    sys.argv = saved[:1]
    try:
        for package in packages:
            for modname in iter_modules(resolve_name(package)):
                try:
                    module = importlib.import_module(modname)
                except Exception:  # noqa: S112, BLE001
                    continue
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if predicate(name, obj):
                        found[name] = obj
    finally:
        sys.argv = saved
    return found


@cache
def load_algorithms():
    """
    :return: dict of available algorithms
    """
    discovery = load_discovery()["algorithms"]

    return discover_classes(
        discovery,
        lambda _, obj: (
            issubclass(obj, BaseAlgorithm)
            and obj is not BaseAlgorithm
            and obj is not OffPolicyAlgorithm
            and obj is not OnPolicyAlgorithm
        ),
    )


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
        return next(a for a in args if a is not type(None))

    return annotation


@cache
def load_env_wrappers():
    """
    :return: dict of available environment wrappers
    """
    discovery = load_discovery()["env_wrappers"]

    wrappers = discover_classes(
        discovery,
        lambda _, obj: (
            issubclass(obj, gymnasium.Wrapper)
            and obj is not gymnasium.Wrapper
            or issubclass(obj, VecEnvWrapper)
            and obj is not VecEnvWrapper
        ),
    )

    return dict(
        sorted(
            wrappers.items(), key=lambda x: 0 if issubclass(x[1], VecEnvWrapper) else 1
        )
    )


def resolve_name(name):
    parts = name.split(".")

    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
            parts = parts[i:]
            break
        except ImportError:
            continue
    else:
        raise ImportError(f"Cannot resolve {name}")

    for part in parts:
        obj = getattr(obj, part)

    return obj


def parse_val(s: str):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def validate_ast(node):
    for child in ast.walk(node):
        if type(child) not in ALLOWED_NODES:
            raise ValueError(f"Unsupported expression: {type(child).__name__}")


def parse_lambda(s):
    """
    Parses a string into a lambda function or a resolved name.
    The lambda is restricted to a subset of Python syntax to prevent arbitrary code execution.

    Useful for stuff like lr where a lambda can be passed.

    :param s: string to parse
    :return: lambda function or resolved name
    """
    if not isinstance(s, str):
        return s

    tree = ast.parse(s, mode="eval")

    if isinstance(tree.body, ast.Lambda):
        validate_ast(tree)
        return eval(compile(tree, "<lambda>", "eval"), {"__builtins__": {}})

    try:
        if isinstance(tree.body, (ast.Name, ast.Attribute)):
            return resolve_name(s)
    except Exception as e:  # noqa: BLE001 failure is expected for stuff like auto
        print(e, file=sys.stderr)
        print(f"Using {s} as is")

    return s


def parse_params(data):
    if isinstance(data, dict):
        return {k: parse_params(v) for k, v in data.items()}
    return parse_lambda(data)
