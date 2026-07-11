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
        with gr.Blocks(delete_cache=(3600, 3600), fill_width=True, fill_height=True) as editor:
            gr.Markdown("# QDrive Training UI")

            with gr.Tab("Train"):
                self.training_tab.build()
            with gr.Tab("Eval"):
                self.eval_tab.build()

        editor.queue()
        editor.launch(theme=gr.Theme.from_hub("JohnSmith9982/small_and_pretty"))
