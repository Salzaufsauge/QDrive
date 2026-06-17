from pathlib import Path

import gradio as gr


class ModelLoader:
    def __init__(self, model_path: Path | str = "."):
        self.model_path = model_path


    def refresh_models(self, current_model):
        choices = [str(p) for p in self.model_path.glob("**/*.zip")]
        return gr.update(choices=choices, value=current_model)

    def load_model(self, model_path):
        pass

    def build_model_loader(self):
        models = [str(p) for p in self.model_path.glob("**/*.zip")]
        with gr.Row():
            model = gr.Dropdown(label="Model", choices=models, value=None, interactive=True, allow_custom_value=True)
            load = gr.Button("Load Model")

        load.click(self.load_model, inputs=[model], outputs=[model])

        timer = gr.Timer(5)
        timer.tick(self.refresh_models, inputs=[model], outputs=[model])
