import inspect
from datetime import datetime

from nicegui import ui
from stable_baselines3.common.env_util import make_vec_env

from backend.config.config import ExperimentConfig
from backend.config.storage import load_config
from util.inspection_helper import load_algorithms, load_env_wrappers, unwrap_optional
from util.utils import replace_empty_strings, parse_val, get_config_path


def build_config(params, sig_params):
    temp = dict()
    for key in sig_params.keys():
        val = unwrap_ui_elem(params.pop(0))
        if val is not None:
            ann = unwrap_optional(sig_params[key].annotation)

            if ann is int:
                val = int(val)

            if key.endswith("kwargs"):
                temp[key] = {key: parse_val(value) for key, value in val.items() if key and value not in [None, ""]}
            else:
                temp[key] = parse_val(val)
    return temp

def build_wrapper_config(params: list, env_wrappers: dict):
    wrappers = load_env_wrappers()
    while True:
        next = params[0]
        if isinstance(next, ui.checkbox):
            trimmed_name = next.text.removeprefix("Enable ")
            if trimmed_name in wrappers:
                if unwrap_ui_elem(params.pop(0)):
                    wrapper_params = inspect.signature(wrappers[trimmed_name]).parameters
                    env_wrappers[trimmed_name] = build_config(params, wrapper_params)
        else:
            break


def unwrap_ui_elem(elem):
    if isinstance(elem, (ui.input, ui.checkbox, ui.number, ui.textarea, ui.select, ui.input_chips)):
        return elem.value
    if isinstance(elem, ui.label):
        return elem.text
    if isinstance(elem, ui.aggrid):
        return {
            row["key"]: row["value"]
            for row in elem.options["rowData"]
        }

    raise ValueError(f"Not a known ui element: {type(elem).__name__}")

class ConfigBuilder:
    @staticmethod
    def write_config(params: list):
        conf = dict()
        try:
            config_path = unwrap_ui_elem(params.pop(0))
            if config_path is not None:
                return load_config(config_path)
            conf["env_param"] = dict()
            env_params = inspect.signature(make_vec_env).parameters
            conf["env_param"] = conf["env_param"] | build_config(params, env_params)
            env_wrappers = conf["env_wrappers"] = dict()
            build_wrapper_config(params, env_wrappers)
            conf["model_param"] = dict()
            conf["algorithm"] = unwrap_ui_elem(params.pop(0))
            conf["milestones"] = sorted(list(unwrap_ui_elem(params.pop(0))))
            conf["total_timesteps"] = unwrap_ui_elem(params.pop(0))
            conf["callback_params"] = dict()
            conf["callback_params"]["eval_freq"] = unwrap_ui_elem(params.pop(0))
            conf["callback_params"]["n_eval_episodes"] = unwrap_ui_elem(params.pop(0))
            conf["callback_params"]["deterministic"] = unwrap_ui_elem(params.pop(0))
            model_params = inspect.signature(load_algorithms()[conf["algorithm"]]).parameters
            conf["model_param"] = conf["model_param"] | build_config(params, model_params)
            conf[
                "model_path"] = f"models/{conf['env_param']['env_id']}/{conf['algorithm']}/model-{conf['model_param']['policy']}-{datetime.now().strftime('%Y-%m-%d_%H-%M')}.zip"
        except Exception as e:
            raise e
        return ExperimentConfig(replace_empty_strings(conf), get_config_path(conf["model_path"]))