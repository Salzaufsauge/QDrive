from pathlib import Path

import gradio as gr

from backend.controller import Controller
from ui.components.eval_tab import EvalTab
from ui.components.training_tab import TrainingTab


class Editor:
    def __init__(self, controller: Controller, model_path: Path):
        self.training_tab = TrainingTab(controller, model_path=model_path)
        self.eval_tab = EvalTab(controller, model_path=model_path)
        self.model_path = model_path

    def launch(self):
        with gr.Blocks() as editor:
            gr.Markdown("# QDrive Training UI")

            self.training_tab.build()
            self.eval_tab.build()

        editor.queue()
        editor.launch(pwa=True)
