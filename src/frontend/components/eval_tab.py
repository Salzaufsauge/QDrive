from pathlib import Path

import gradio as gr

from backend.config.storage import load_config
from backend.controller import Controller
from frontend.components.config_loader import ConfigLoader


class EvalTab:
    def __init__(self, controller: Controller, config_path: Path):
        self.controller = controller
        self.config = None
        self.config_path = config_path

    def start_eval(self):
        if self.config.model_path is None:
            raise gr.Error("No model loaded")
        yield from self.controller.start_eval(self.config)

    def stop_eval(self):
        self.controller.stop_eval()
        yield [
            gr.update(visible=False),
            gr.update(visible=True),
        ]

    def load_model(self, config_path: Path):
        self.config = load_config(config_path)
        model_path = self.config.abs_model_path
        if not model_path.exists():
            raise gr.Error(f"Model {model_path} not found")
        return gr.update(visible=True, value=f"Model {model_path} loaded")

    def build(self):
        with gr.Tab("Eval"):
            model_loader = ConfigLoader(self.config_path)
            model_loader.build_config_loader()
            model_loader.load_btn.click(self.load_model, inputs=[model_loader.config],
                                        outputs=[model_loader.load_label])

            eval_btn = gr.Button("Evaluate")
            stop_btn = gr.Button("Stop", visible=False)

            output = gr.Image(interactive=False, streaming=True, type="numpy", label="Output")

            eval_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)),
                           outputs=[eval_btn, stop_btn]).then(self.start_eval, outputs=output)
            stop_btn.click(self.stop_eval, outputs=[stop_btn, eval_btn])
