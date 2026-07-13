import collections.abc
import importlib
import inspect
import numbers
import pkgutil
import types
import typing
import uuid
from functools import cache
from typing import get_origin

import gymnasium
import yaml
from nicegui import ui
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnvWrapper

import util.utils


@cache
def load_discovery():
    return yaml.safe_load((util.utils.get_project_root() / "configs/discovery.yaml").read_text())

def iter_modules(package):
    for _, modname, _ in pkgutil.walk_packages(
            package.__path__, package.__name__ + "."
    ):
        yield modname


def discover_classes(packages, predicate):
    found = {}

    for package in packages:
        for modname in iter_modules(resolve_name(package)):
            try:
                module = importlib.import_module(modname)
            except Exception:
                continue
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if predicate(name, obj):
                    found[name] = obj
    return found


def add_table(table_name, table_data):
    for row in table_data:
        row["_id"] = str(uuid.uuid4())

    with ui.column().classes('flex-grow'):
        ui.label(table_name).classes('text-lg font-bold')
        grid = ui.aggrid(
            {
                'columnDefs': [
                    {"headerName": "_id", "field": "_id", 'hide': True},
                    {"name": "key", "label": "key", "field": "key", 'editable': True},
                    {"name": "value", "label": "value", "field": "value", 'editable': True},
                ],
                'rowData': table_data,
                'rowSelection': 'multiple',
                'stopEditingWhenCellsLoseFocus': True,
            },
            auto_size_columns=True,
        ).classes('flex-grow')
        with ui.row().classes('flex-grow'):
            add_btn = ui.button(f'Add Row').classes('flex-grow')
            rm_btn = ui.button(f"Remove selected Rows").classes('flex-grow')

    def add_row():
        new_id = str(uuid.uuid4())
        grid.options['rowData'].append({'_id': new_id, 'key': None, 'value': None})

    def handle_cell_value_change(e):
        new_row = e.args['data']
        grid.options['rowData'][:] = [row | new_row if row['_id'] ==
                                                       new_row['_id'] else row for row in grid.options['rowData']]

    async def delete_selected():
        selected_id = [row['_id'] for row in await grid.get_selected_rows()]
        grid.options['rowData'][:] = [row for row in grid.options['rowData'] if row['_id'] not in selected_id]

    grid.on('cellValueChanged', handle_cell_value_change)
    add_btn.on_click(add_row)
    rm_btn.on_click(delete_selected)

    return grid


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
            elem = ui.input(
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
            value=val,
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

        elem = add_table(param.name, rows)

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
            value=val,
        )

    elif param.name.endswith("keys"):
        rows = [
            {"key": k, "value": v}
            for k, v in (val or {}).items()
        ]

        elem = add_table(param.name, rows)

    else:
        elem = ui.input(
            label=f"{param.name} (unknown type)",
            value=str(val) if val is not None else ''
        )

    elem.set_visibility(visible)
    return elem.classes('flex-grow')


@cache
def load_algorithms():
    discovery = load_discovery()["algorithms"]

    return discover_classes(discovery, lambda _, obj:
    issubclass(obj, BaseAlgorithm)
    and obj is not BaseAlgorithm
    and obj is not OffPolicyAlgorithm
    and obj is not OnPolicyAlgorithm
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
        return [a for a in args if a is not type(None)][0]

    return annotation


@cache
def load_env_wrappers():
    discovery = load_discovery()["env_wrappers"]

    wrappers = discover_classes(discovery, lambda _, obj:
    issubclass(obj, gymnasium.Wrapper)
    and obj is not gymnasium.Wrapper or
    issubclass(obj, VecEnvWrapper)
    and obj is not VecEnvWrapper
                                )

    return dict(sorted(wrappers.items(), key=lambda x: 0 if issubclass(x[1], VecEnvWrapper) else 1))


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
