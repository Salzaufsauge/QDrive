import inspect
from ast import literal_eval
from datetime import datetime
from pathlib import Path

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from backend.config.config import ExperimentConfig
from backend.config.storage import load_config
from util.inspection_helper import load_algorithms
from util.utils import replace_empty_strings


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
        return literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def build_config(params, sig_params):
    temp = dict()
    for key in sig_params.keys():
        val = params.pop(0)
        if val is not None:
            if key.endswith("kwargs"):
                temp[key] = {row[0]: parse_val(row[1]) for row in val if row and row[1] not in [None, ""]}
            else:
                temp[key] = parse_val(val)
    return temp


class ConfigBuilder:
    def __init__(self):
        self.algorithms = load_algorithms()

    @staticmethod
    def write_config(self, params):
        conf = dict()
        try:
            config_path = params.pop(0)
            if config_path is not None:
                return load_config(config_path)
            conf["env_param"] = dict()
            env_params = inspect.signature(make_vec_env).parameters
            conf["env_param"] = conf["env_param"] | self.__build_config__(params, env_params)
            conf["vec_frame_stack"] = dict()
            conf["vec_frame_stack"]["enabled"] = params.pop(0)
            vec_frame_stack_params = inspect.signature(VecFrameStack).parameters
            conf["vec_frame_stack"] = conf["vec_frame_stack"] | self.__build_config__(params, vec_frame_stack_params)
            conf["model_param"] = dict()
            conf["algorithm"] = params.pop(0)
            conf["milestones"] = sorted(list(params.pop(0)))
            conf["total_timesteps"] = params.pop(0)
            model_params = inspect.signature(self.algorithms[conf["algorithm"]]).parameters
            conf["model_param"] = conf["model_param"] | self.__build_config__(params, model_params)
            conf[
                "model_path"] = f"models/{conf['env_param']['env_id']}/{conf['algorithm']}/model-{conf['model_param']['policy']}-{datetime.now().strftime('%Y-%m-%d_%H-%M')}.zip"
        except Exception as e:
            raise e
        return ExperimentConfig(replace_empty_strings(conf), get_config_path(conf["model_path"]))
