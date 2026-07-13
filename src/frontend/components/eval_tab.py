import threading
from pathlib import Path

from fastapi import Response
from nicegui import ui, app, run

from backend.config.storage import load_config
from backend.controller import Controller
from frontend.components.config_loader import ConfigLoader
from util.utils import frame_to_data_url


class EvalTab:
    def __init__(self, controller: Controller, config_path: Path):
        self.config_loader = None
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
        threading.Thread(target=self._eval_loop).start()

    def _eval_loop(self):
        try:

            for _ in self.controller.start_eval(self.config):
                if not self.running:
                    break
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
            self.config_loader.load_label.set_visibility(True)

        except Exception as e:
            ui.notify(
                str(e),
                type="negative"
            )

    def build(self):

        self.config_loader = ConfigLoader(
            self.config_path
        )

        self.config_loader.build_config_loader()

        self.config_loader.load_btn.on_click(
            lambda: self.load_config(self.config_loader.config.value)
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

        @app.get('/video/frame')
        async def get_frame() -> Response:
            frame = self.controller.get_current_frame()
            if frame is None:
                print("No frame available")
                return Response(status_code=204)
            jpeg = await run.cpu_bound(frame_to_data_url, frame)
            return Response(jpeg, media_type='image/jpeg')

        with ui.row().classes('w-full justify-center'):
            self.output = ui.interactive_image('video/frame').classes('w-[640px] h-[480px] object-contain')
            ui.timer(0.1, self.output.force_reload)
