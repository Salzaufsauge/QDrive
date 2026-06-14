from pathlib import Path

import gradio as gr

from ui.components.eval_tab import EvalTab
from ui.components.training_tab import TrainingTab

class Editor:
    def __init__(self, model_path: Path | str = "."):
        self.training_tab = TrainingTab(model_path=model_path)
        self.eval_tab = EvalTab()
        self.model_path = model_path

    def launch(self):
        with gr.Blocks() as editor:
            gr.Markdown("# QDrive Training UI")

            self.training_tab.build()
            self.eval_tab.build()


        editor.queue()
        editor.launch(pwa=True)
