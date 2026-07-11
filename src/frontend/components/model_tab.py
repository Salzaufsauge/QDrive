import inspect
from functools import partial

import gradio as gr

from frontend.components import TagComponent
from util.inspection_helper import load_algorithms, get_policies_from_algo, make_ui_for_param
from util.utils import build_ui_params


def make_model_ui(param: inspect.Parameter, algorithms, algo):
    if param.name == "policy":
        policy = get_policies_from_algo(algorithms[algo]).keys()
        return gr.Dropdown(choices=policy, label=param.name, interactive=True)
    if param.name == "env":
        return gr.Label(value=None, visible=False)
    if param.name == "tensorboard_log":  # always proj_root/logs
        return gr.Label(value=None, visible=False)
    else:
        return make_ui_for_param(param)

class ModelTab:
    def __init__(self):
        self.algorithms = load_algorithms()
        self.model_params = list()
        self.model_params_base_len: int = 0

    def build(self):
        self.model_params.append(
            gr.Dropdown(value="PPO", choices=self.algorithms.keys(), label="algorithm", interactive=True))
        self.model_params.append(TagComponent.TagComponent())
        self.model_params.append(gr.Number(label="total_timesteps", value=1000000, interactive=True))
        with gr.Accordion("Callback Parameters"):
            with gr.Row():
                self.model_params.append(gr.Number(label="eval_freq", value=10000, interactive=True))
                self.model_params.append(gr.Number(label="n_eval_episodes", value=10, interactive=True))
                self.model_params.append(gr.Checkbox(label="deterministic", value=True, interactive=True))
        self.model_params_base_len = len(self.model_params)

    def get_model_params(self, algo):
        params = list(inspect.signature(self.algorithms[algo]).parameters.values())

        temp = build_ui_params(params, 4, partial(make_model_ui, algorithms=self.algorithms, algo=algo))

        self.model_params[self.model_params_base_len:] = temp
        return self.model_params
