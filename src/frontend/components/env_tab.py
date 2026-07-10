import inspect

import gradio as gr
from stable_baselines3.common.env_util import make_vec_env

from util.inspection_helper import make_ui_for_param
from util.utils import get_envs


class EnvTab:
    def __init__(self):
        self.env_params = list()

    def build(self):
        params = list(inspect.signature(make_vec_env).parameters.values())
        for i in range(0, len(params), 3):
            with gr.Row():
                for param in params[i:i + 3]:
                    if param.name == "env_id":
                        self.env_params.append(
                            gr.Dropdown(value="CarRacing-v3", choices=get_envs(), label=param.name,
                                        interactive=True))
                        continue
                    if param.name == "vec_env_cls":
                        self.env_params.append(
                            gr.Dropdown(value="DummyVecEnv", choices=["DummyVecEnv", "SubprocVecEnv"],
                                        label=param.name, interactive=True))
                        continue
                    if param.name == "wrapper_class":
                        self.env_params.append(gr.Label(value=None, visible=False))
                        continue
                    self.env_params.append(make_ui_for_param(param))

    def get_env_params(self):
        return self.env_params
