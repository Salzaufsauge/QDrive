from pathlib import Path

from nicegui import ui

from backend.config.storage import load_config
from backend.controller import Controller
from frontend.components.config_loader import ConfigLoader


class EvalTab:
    def __init__(self, controller: Controller, config_path: Path):
        self.model_loader_label = None
        self.controller = controller
        self.config = None
        self.config_path = config_path

        self.eval_btn = None
        self.stop_btn = None
        self.output = None

        self.running = False

    def start_eval(self):
        self.eval_btn.set_visibility(False)
        self.stop_btn.set_visibility(True)

        if self.config is None or self.config.model_path is None:
            ui.notify(
                "No model loaded",
                type="negative"
            )
            self.stop_eval()
            return

        self.running = True

        try:
            for frame in self.controller.start_eval(self.config):
                if not self.running:
                    break

                self.output.set_source(frame)
        except Exception as e:
            ui.notify(
                str(e),
                type="negative"
            )
            self.stop_eval()
            return

    def stop_eval(self):
        self.controller.stop_eval()
        self.running = False

        self.stop_btn.set_visibility(False)
        self.eval_btn.set_visibility(True)

    def load_config(self, config_path):
        try:
            self.config = load_config(
                config_path
            )

            model_path = self.config.abs_model_path

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model {model_path} not found"
                )

            self.model_loader_label.set_text(
                f"Model {model_path} loaded"
            )
            self.model_loader_label.set_visibility(True)

        except Exception as e:
            ui.notify(
                str(e),
                type="negative"
            )

    def build(self):

        model_loader = ConfigLoader(
            self.config_path
        )

        model_loader.build_config_loader()

        model_loader.load_btn.on_click(
            lambda: self.load_config(model_loader.config.value)
        )

        self.model_loader_label = ui.label("")
        self.model_loader_label.set_visibility(False)

        with ui.row().classes("w-full"):
            self.eval_btn = ui.button(
                "Evaluate",
                on_click=self.start_eval
            ).classes("flex-grow")

            self.stop_btn = ui.button(
                "Stop",
                on_click=self.stop_eval
            ).classes("flex-grow")

            self.stop_btn.set_visibility(False)

        self.output = ui.image()