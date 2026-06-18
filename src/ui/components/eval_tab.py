from pathlib import Path

import gradio as gr

from backend.configuration import Configuration
from backend.controller import Controller
from ui.components.model_loader import ModelLoader


class EvalTab:
    def __init__(self, controller: Controller, model_path: Path | str = ".", ):
        self.controller = controller
        self.config = Configuration()
        self.model_path = model_path
        self.running = False

    def start_eval(self):
        yield from self.controller.start_eval(self.config)

    def stop_eval(self):
        self.controller.stop_eval()
        yield [
            gr.update(visible=False),
            gr.update(visible=True),
        ]

    def load_model(self, model_path: Path):
        self.config.load_model(model_path)
        return gr.update(visible=True, value=f"Model {model_path} loaded")

    def build(self):
        with gr.Tab("Eval"):
            model_loader = ModelLoader(self.model_path)
            model_loader.build_model_loader()
            model_loader.load_btn.click(self.load_model, inputs=[model_loader.model], outputs=[model_loader.load_label])

            eval_btn = gr.Button("Evaluate")
            stop_btn = gr.Button("Stop", visible=False)

            output = gr.Image(interactive=False, streaming=True, type="pil", label="Output")

            eval_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)),
                           outputs=[eval_btn, stop_btn]).then(self.start_eval, outputs=output)
            stop_btn.click(self.stop_eval, outputs=[stop_btn, eval_btn])
