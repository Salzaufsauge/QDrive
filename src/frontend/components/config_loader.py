from pathlib import Path

import gradio as gr


class ConfigLoader:
    def __init__(self, config_path: Path):
        self.load_label = None
        self.load_btn = None
        self.config = None
        self.config_path = config_path


    def refresh_models(self, current_model):
        choices = [str(p.relative_to(self.config_path.parent)) for p in self.config_path.glob("**/*.yaml")]
        return gr.update(choices=choices, value=current_model)

    def build_config_loader(self):
        models = [str(p.relative_to(self.config_path.parent)) for p in self.config_path.glob("**/*.yaml")]
        with gr.Row(equal_height=True):
            self.config = gr.Dropdown(label="Model", choices=models, value=None, interactive=True)
            with gr.Column():
                self.load_btn = gr.Button("Load Model")
                self.load_label = gr.Label(value="Model Loaded", visible=False)

        timer = gr.Timer(5)
        timer.tick(self.refresh_models, inputs=[self.config], outputs=[self.config])
