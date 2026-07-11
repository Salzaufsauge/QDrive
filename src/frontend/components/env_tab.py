import inspect

import gradio as gr
from stable_baselines3.common.env_util import make_vec_env

from util.inspection_helper import make_ui_for_param
from util.utils import get_envs, build_ui_params


def make_env_ui(param: inspect.Parameter):
    if param.name == "env_id":
        return gr.Dropdown(value="CarRacing-v3", choices=get_envs(), label=param.name,
                           interactive=True)
    if param.name == "vec_env_cls":
        return gr.Dropdown(value="DummyVecEnv", choices=["DummyVecEnv", "SubprocVecEnv"],
                           label=param.name, interactive=True)
    if param.name == "wrapper_class":
        return gr.Label(value=None, visible=False)
    else:
        return make_ui_for_param(param)

class EnvTab:
    def __init__(self):
        self.env_params = list()

    def build(self):
        params = list(inspect.signature(make_vec_env).parameters.values())

        self.env_params.extend(build_ui_params(params, 4, make_env_ui))


    def get_env_params(self):
        return self.env_params
