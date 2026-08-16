from functools import partial
from pathlib import Path

from nicegui import ui
from yaml import YAMLError

from backend.config.storage import load_config
from backend.controller import Controller
from frontend.components.config_loader import ConfigLoader


class EvalTab:
    def __init__(self, controller: Controller, config_path: Path):
        self.model_loader_label = None
        self.config_loader = None
        self.controller = controller
        self.config = None
        self.config_path = config_path

        self.eval_btn = None
        self.stop_btn = None

    async def start_eval(self, eval_timer: ui.timer):
        self.eval_btn.set_visibility(False)
        self.stop_btn.set_visibility(True)

        if self.config is None or self.config.model_path is None:
            ui.notify("No model loaded", type="negative")
            await self.stop_eval(eval_timer)
            return
        self.controller.start_eval(self.config)
        eval_timer.activate()

    async def stop_eval(self, eval_timer: ui.timer):
        await self.controller.stop_eval()

        eval_timer.deactivate()

        self.stop_btn.set_visibility(False)
        self.eval_btn.set_visibility(True)

    def load_config(self, config_path):
        if config_path is None and self.config is not None:
            self.config = None
            ui.notify("Unloaded config")
            return
        if config_path is None:
            ui.notify("No config selected")
            return
        try:
            self.config = load_config(config_path)

            model_path = self.config.abs_model_path

            if not model_path.exists():
                raise FileNotFoundError(f"Model {model_path} not found")
            self.config_loader.load_label.set_visibility(True)

        except FileNotFoundError as e:
            ui.notify(str(e), type="negative")
        except KeyError as e:
            ui.notify(f"Missing key in config: {e}", type="negative")
        except YAMLError as e:
            ui.notify(f"Error parsing config: {e}", type="negative")

    def build(self):

        self.config_loader = ConfigLoader(self.config_path)

        self.config_loader.build_config_loader()

        self.config_loader.load_btn.on_click(
            lambda: self.load_config(self.config_loader.config.value)
        )

        self.model_loader_label = ui.label("")
        self.model_loader_label.set_visibility(False)

        with ui.row().classes("w-full"):
            self.eval_btn = ui.button("Evaluate").classes("flex-grow")
            self.stop_btn = ui.button("Stop").classes("flex-grow")

            self.stop_btn.set_visibility(False)

        with ui.row().classes("w-full justify-center"):
            output = ui.interactive_image("video/frame").classes(
                "w-[640px] h-[480px] object-contain"
            )

        eval_timer = ui.timer(0.017, output.force_reload, active=False)

        self.eval_btn.on_click(partial(self.start_eval, eval_timer))
        self.stop_btn.on_click(partial(self.stop_eval, eval_timer))

        def handle_on_delete():
            eval_timer.cancel()

        ui.context.client.on_delete(handle_on_delete)
