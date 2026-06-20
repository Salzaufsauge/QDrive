import datetime
import inspect
from pathlib import Path

import yaml
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from util.inspection_helper import load_algorithms
from util.utils import get_project_root, replace_empty_strings


def get_config_path(model_path):
    model_path = Path(model_path)
    parts = list(model_path.parts)

    if "models" in parts:
        i = parts.index("models")
        parts[i] = "experiments"

    conf_path = Path(*parts)
    conf_path = conf_path.with_suffix(".yaml")

    return conf_path


def __build_config__(params, sig_params):
    temp = dict()
    for key in sig_params.keys():
        val = params.pop(0)
        if val is not None:
            if key.endswith("kwargs"):
                temp[key] = {row[0]: row[1] for row in val if row and row[1] not in [None, ""]}
            else:
                temp[key] = val
    return temp

class Configuration:
    def __init__(self):
        self.config = dict()
        self.algorithms = load_algorithms()

    def write_config(self, params):
        conf = dict()
        try:
            model_path = params.pop(0)
            if model_path is not None:
                self.load_model(model_path)
                return
            conf["env_param"] = dict()
            env_params = inspect.signature(make_vec_env).parameters
            conf["env_param"] = conf["env_param"] | __build_config__(params, env_params)
            conf["vec_frame_stack"] = dict()
            conf["vec_frame_stack"]["enabled"] = params.pop(0)
            vec_frame_stack_params = inspect.signature(VecFrameStack).parameters
            conf["vec_frame_stack"] = conf["vec_frame_stack"] | __build_config__(params, vec_frame_stack_params)
            conf["model_param"] = dict()
            conf["algorithm"] = params.pop(0)
            model_params = inspect.signature(self.algorithms[conf["algorithm"]]).parameters
            conf["model_param"] = conf["model_param"] | __build_config__(params, model_params)
            conf["total_timesteps"] = params.pop(0)
            conf[
                "model_path"] = f"models/{conf['env_param']['env_id']}/{conf['algorithm']}/model-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip"
        except Exception as e:
            raise e
        self.config = replace_empty_strings(conf)

    def load_model(self, model_path: Path):
        self.config = yaml.safe_load(get_config_path(get_project_root() / model_path).read_text())

    def save_model(self, model):
        model_path = get_project_root() / self.config["model_path"]
        model.save(model_path)
        cfg = yaml.safe_dump(self.config, sort_keys=False)
        cfg_path = get_config_path(model_path)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with cfg_path.open("w") as f:
            f.write(cfg)
