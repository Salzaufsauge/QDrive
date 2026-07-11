import threading
from pathlib import Path

from nicegui import ui

from backend.config.builder import ConfigBuilder
from backend.config.storage import load_config
from backend.controller import Controller
from frontend.components.config_loader import ConfigLoader
from frontend.components.env_tab import EnvTab
from frontend.components.model_tab import ModelTab
from frontend.components.wrapper_tab import WrapperTab


class TrainingTab:
    def __init__(self, controller: Controller, config_path: Path):
        self.config_loader = None
        self.controller = controller
        self.config_path = config_path
        self.config = None

        self.env_tab = None
        self.wrapper_tab = None
        self.model_tab = None

        self.train_btn = None
        self.stop_btn = None

        self.console = None
        self.graph = None

        self.model_container = None

    def setup_config(self, *params):
        try:
            self.config = ConfigBuilder.write_config(list(params))
        except Exception as e:
            ui.notify(f"Error: {e}", type="negative")
            raise

    def start_training(self):
        thread = threading.Thread(
            target=self.controller.start_training,
            args=(self.config,),
            daemon=True,
        )
        thread.start()

    def get_training_state(self):
        for state in self.controller.get_training_state():
            if self.console:
                self.console.value = state

    def stop_training(self):
        self.controller.stop_training()

        self.stop_btn.set_visibility(False)
        self.train_btn.set_visibility(True)

    def load_config(self):
        try:
            self.config = load_config(self.config_path)

        except Exception as e:
            ui.notify(f"Error: {e}", type="negative")
            return

        self.config_label.set_text(
            f"Config {self.config_path} loaded"
        )
        self.config_label.set_visibility(True)

    def build(self):
        self.config_loader = ConfigLoader(self.config_path)
        self.config_loader.build_config_loader()
        self.config_loader.load_btn.on_click(
            lambda: self.load_config()
        )

        with ui.tabs().classes('w-full justify-center') as tabs:
            env_tab_btn = ui.tab("Environment Parameters")
            wrapper_tab_btn = ui.tab("Wrapper Parameters")
            model_tab_btn = ui.tab("Model Parameters")

        with ui.tab_panels(tabs, value=env_tab_btn).classes('w-full flex-grow'):
            with ui.tab_panel(env_tab_btn):
                self.env_tab = EnvTab()
                self.env_tab.build()

            with ui.tab_panel(wrapper_tab_btn):
                self.wrapper_tab = WrapperTab()
                self.wrapper_tab.build()

            with ui.tab_panel(model_tab_btn):
                self.model_tab = ModelTab()
                self.model_tab.build()

                self.model_container = ui.column().classes('w-full')

                def update_model_params(e):
                    self.model_container.clear()

                    with self.model_container:
                        self.model_tab.get_model_params(
                            e.value
                        )

                update_model_params(self.model_tab.model_params[0])

                self.model_tab.model_params[0].on_value_change(
                    update_model_params
                )

        self.train_btn = ui.button(
            "Train",
            on_click=self.train
        ).classes('w-full')

        self.stop_btn = ui.button(
            "Stop",
            on_click=self.stop_training
        ).classes('w-full')

        self.stop_btn.set_visibility(False)

        self.config_label = ui.label("")
        self.config_label.set_visibility(False)

        with ui.expansion(
                "Training Output",
                value=True
        ).classes('w-full'):
            with ui.row().classes('w-full'):
                self.console = (
                    ui.textarea(label="Console")
                    .props("readonly")
                    .classes("flex-1")
                    .style("height: 400px")
                )

                self.graph = (
                    ui.plotly({})
                    .classes("flex-1")
                    .style("height: 400px")
                )

    def train(self):
        try:
            self.setup_config(
                *self.config_loader.config.value,
                *self.env_tab.get_env_params(),
                *self.wrapper_tab.wrapper_params,
                *self.model_tab.model_params,
            )

        except Exception:
            return

        self.train_btn.set_visibility(False)
        self.stop_btn.set_visibility(True)

        self.start_training()
