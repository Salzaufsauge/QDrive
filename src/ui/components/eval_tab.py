from pathlib import Path

import gradio as gr

from ui.components.model_loader import ModelLoader
from util.gym_helper import get_envs


class EvalTab:
    def __init__(self, model_path: Path | str = "."):
        self.model_path = model_path
        self.running = False

    def start_eval(self):
        yield [
            gr.update(visible=False),
            gr.update(visible=True),
        ]

    def stop_eval(self):
        self.running = False
        yield [
            gr.update(visible=False),
            gr.update(visible=True),
        ]

    def build(self):
        with gr.Tab("Eval"):
            ModelLoader(self.model_path).build_model_loader()

            env_id = gr.Dropdown(value="CarRacing-v3", choices=get_envs(), label="Environment")

            train = gr.Button("Train")
            stop = gr.Button("Stop", visible=False)

            gr.Image(interactive=False, streaming=True, type="pil", label="Output")

            train.click(self.start_eval, outputs=[train, stop])
            stop.click(self.stop_eval, outputs=[stop, train])
