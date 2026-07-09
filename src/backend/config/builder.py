import inspect
from datetime import datetime

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from backend.config.config import ExperimentConfig
from backend.config.storage import load_config
from util.utils import replace_empty_strings, parse_val, get_config_path


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
    @staticmethod
    def write_config(params: list, algorithms):
        conf = dict()
        try:
            config_path = params.pop(0)
            if config_path is not None:
                return load_config(config_path)
            conf["env_param"] = dict()
            env_params = inspect.signature(make_vec_env).parameters
            conf["env_param"] = conf["env_param"] | build_config(params, env_params)
            env_wrappers = conf["env_wrappers"] = list()
            vec_frame_stack_params = inspect.signature(VecFrameStack).parameters
            if params.pop(0):
                env_wrappers.append({"VecFrameStack": build_config(params, vec_frame_stack_params)})
            else:
                for _ in vec_frame_stack_params.values():
                    params.pop(0)
            conf["model_param"] = dict()
            conf["algorithm"] = params.pop(0)
            conf["milestones"] = sorted(list(params.pop(0)))
            conf["total_timesteps"] = params.pop(0)
            conf["callback_params"] = dict()
            conf["callback_params"]["eval_freq"] = params.pop(0)
            conf["callback_params"]["n_eval_episodes"] = params.pop(0)
            conf["callback_params"]["deterministic"] = params.pop(0)
            model_params = inspect.signature(algorithms[conf["algorithm"]]).parameters
            conf["model_param"] = conf["model_param"] | build_config(params, model_params)
            conf[
                "model_path"] = f"models/{conf['env_param']['env_id']}/{conf['algorithm']}/model-{conf['model_param']['policy']}-{datetime.now().strftime('%Y-%m-%d_%H-%M')}.zip"
        except Exception as e:
            raise e
        return ExperimentConfig(replace_empty_strings(conf), get_config_path(conf["model_path"]))
