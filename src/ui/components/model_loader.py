from pathlib import Path

import gradio as gr


class ModelLoader:
    def __init__(self, model_path: Path | str = "."):
        self.load_label = None
        self.load_btn = None
        self.model = None
        self.model_path = model_path


    def refresh_models(self, current_model):
        choices = [str(p.relative_to(self.model_path.parent)) for p in self.model_path.glob("**/*.zip")]
        return gr.update(choices=choices, value=current_model)

    def build_model_loader(self):
        models = [str(p.relative_to(self.model_path.parent)) for p in self.model_path.glob("**/*.zip")]
        with gr.Row(equal_height=True):
            self.model = gr.Dropdown(label="Model", choices=models, value=None, interactive=True)
            with gr.Column():
                self.load_btn = gr.Button("Load Model")
                self.load_label = gr.Label(value="Model Loaded", visible=False)

        timer = gr.Timer(5)
        timer.tick(self.refresh_models, inputs=[self.model], outputs=[self.model])
