from pathlib import Path

import gradio as gr

from backend.controller import Controller
from frontend.components.eval_tab import EvalTab
from frontend.components.training_tab import TrainingTab


class Editor:
    def __init__(self, controller: Controller, config_path: Path):
        self.training_tab = TrainingTab(controller, config_path=config_path)
        self.eval_tab = EvalTab(controller, config_path=config_path)
        self.config_path = config_path

    def launch(self):
        with gr.Blocks() as editor:
            gr.Markdown("# QDrive Training UI")

            self.training_tab.build()
            self.eval_tab.build()

        editor.queue()
        editor.launch(pwa=True)
