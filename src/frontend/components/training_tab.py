import threading
from pathlib import Path

import gradio as gr

from backend.config.builder import ConfigBuilder
from backend.config.storage import load_config
from backend.controller import Controller
from frontend.components.config_loader import ConfigLoader
from frontend.components.env_tab import EnvTab
from frontend.components.model_tab import ModelTab
from frontend.components.wrapper_tab import WrapperTab


class TrainingTab:
    def __init__(self, controller: Controller, config_path: Path):
        self.controller = controller
        self.config_path = config_path
        self.config = None

    def setup_config(self, *params):
        try:
            self.config = ConfigBuilder.write_config(list(params))
        except Exception as e:
            raise gr.Error(f"Error: {e}")

    def start_training(self):
        thread = threading.Thread(
            target=self.controller.start_training,
            args=(self.config,),
            daemon=True,
        )
        thread.start()

    def get_training_state(self):
        yield from self.controller.get_training_state()

    def stop_training(self):
        self.controller.stop_training()
        yield [
            gr.update(visible=False),
            gr.update(visible=True),
        ]

    def load_config(self, config_path: Path):
        try:
            self.config = load_config(config_path)
        except Exception as e:
            raise gr.Error(f"Error: {e}")
        return gr.update(visible=True, value=f"Config {config_path} loaded")

    def build(self):
        config_loader = ConfigLoader(self.config_path)
        config_loader.build_config_loader()
        config_loader.load_btn.click(self.load_config, inputs=[config_loader.config],
                                     outputs=config_loader.load_label, )
        with gr.Tab("Environment Parameters"):
            env_tab = EnvTab()
            env_tab.build()
        with gr.Tab("Wrapper Parameters"):
            wrapper_tab = WrapperTab()
            wrapper_tab.build()
        with gr.Tab("Model Parameters"):
            model_tab = ModelTab()
            model_tab.build()

            @gr.render(inputs=model_tab.model_params[0])
            def render(algo):
                model_params = model_tab.get_model_params(algo)
                train.click(self.setup_config,
                            inputs=[config_loader.config] + env_tab.env_params + wrapper_tab.wrapper_params + model_params,
                            ).then(lambda: (gr.update(visible=False), gr.update(visible=True)),
                                   outputs=[train, stop]
                                   ).then(self.start_training).then(self.get_training_state,
                                                                    outputs=[console, graph])

        train = gr.Button("Train")
        stop = gr.Button("Stop", visible=False)

        with gr.Accordion("Training Output", open=True):
            with gr.Row():
                console = gr.Textbox(
                    label="Console",
                    lines=15,
                    interactive=False
                )
                graph = gr.Plot(label="Training Curve")

        stop.click(self.stop_training, outputs=[stop, train])
