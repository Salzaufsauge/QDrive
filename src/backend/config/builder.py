import inspect
import typing

from stable_baselines3.common.env_util import make_vec_env

from backend.config.config import ExperimentConfig
from util.inspection_helper import (
    load_algorithms,
    load_env_wrappers,
    parse_val,
    unwrap_optional,
)
from util.utils import get_config_path, make_model_path, replace_empty_strings


def build_config(params: dict, sig_params) -> dict:
    out = {}
    for name in sig_params:
        val = params.get(name)
        if val is None:
            continue

        ann = unwrap_optional(sig_params[name].annotation)

        if ann is int:
            val = int(val)
        elif int in typing.get_args(ann):
            try:
                val = int(val)
            except (TypeError, ValueError):
                pass

        if isinstance(val, dict):
            out[name] = {
                key: parse_val(value)
                for key, value in val.items()
                if key and value not in [None, ""]
            }
        else:
            out[name] = parse_val(val)
    return out


class ConfigBuilder:
    @staticmethod
    def build(raw: dict):
        wrappers = load_env_wrappers()
        try:
            algo = raw["algorithm"]
            conf = raw | {
                "env_param": build_config(
                    raw["env_param"], inspect.signature(make_vec_env).parameters
                ),
                "env_wrappers": {
                    name: build_config(
                        values, inspect.signature(wrappers[name]).parameters
                    )
                    for name, values in raw["env_wrappers"].items()
                },
                "model_param": build_config(
                    raw["model_param"],
                    inspect.signature(load_algorithms()[algo]).parameters,
                ),
                "milestones": sorted(raw["milestones"], key=int)
                if raw["milestones"]
                else None,
            }
            conf["model_path"] = make_model_path(
                conf["env_param"]["env_id"], algo, conf["model_param"]["policy"]
            )
        except Exception as e:
            raise RuntimeError(f"Error while building config: {e}") from e
        return ExperimentConfig(
            replace_empty_strings(conf), get_config_path(conf["model_path"])
        )
