from pathlib import Path

import gradio as gr

from util.gym_helper import get_envs


class EvalTab:
    def __init__(self, model_path: Path | str = "."):
        self.model_path = model_path

    def build(self):
        with gr.Tab("Eval"):
            env_id = gr.Dropdown(value="CarRacing-v3", choices=get_envs(), label="Environment")
