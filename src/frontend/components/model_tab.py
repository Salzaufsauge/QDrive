import inspect

import gradio as gr

from frontend.components import TagComponent
from util.inspection_helper import load_algorithms, get_policies_from_algo, make_ui_for_param


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
        temp = list()
        params = list(inspect.signature(self.algorithms[algo]).parameters.values())

        for i in range(0, len(params), 3):
            with gr.Row():
                for param in params[i:i + 3]:
                    if param.name == "policy":
                        policy = get_policies_from_algo(self.algorithms[algo]).keys()
                        temp.append(gr.Dropdown(choices=policy, label=param.name, interactive=True))
                        continue
                    if param.name == "env":
                        temp.append(gr.Label(value=None, visible=False))
                        continue
                    if param.name == "tensorboard_log":  # always proj_root/logs
                        temp.append(gr.Label(value=None, visible=False))
                        continue
                    temp.append(make_ui_for_param(param))

        self.model_params[self.model_params_base_len:] = temp
        return self.model_params
