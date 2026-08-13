import base64
from pathlib import Path

from fastapi import Response
from nicegui import app, run, ui

from backend.config.storage import load_config
from backend.controller import Controller
from frontend.components.config_loader import ConfigLoader
from util.utils import frame_to_data_url

black_1px = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII="
placeholder = Response(
    content=base64.b64decode(black_1px.encode("ascii")), media_type="image/png"
)


class EvalTab:
    def __init__(self, controller: Controller, config_path: Path):
        self.model_loader_label = None
        self.config_loader = None
        self.controller = controller
        self.config = None
        self.config_path = config_path
        self.eval_timer = None

        self.eval_btn = None
        self.stop_btn = None
        self.output = None

    async def start_eval(self):
        self.eval_btn.set_visibility(False)
        self.stop_btn.set_visibility(True)

        if self.config is None or self.config.model_path is None:
            ui.notify("No model loaded", type="negative")
            await self.stop_eval()
            return
        self.controller.start_eval(self.config)
        self.eval_timer = ui.timer(0.01, self.output.force_reload)

    async def stop_eval(self):
        await self.controller.stop_eval()

        if self.eval_timer is not None:
            self.eval_timer.cancel()

        self.stop_btn.set_visibility(False)
        self.eval_btn.set_visibility(True)

    def load_config(self, config_path):
        if config_path is None and self.config is not None:
            self.config = None
            ui.notify("Unloaded config")
            return
        try:
            self.config = load_config(config_path)

            model_path = self.config.abs_model_path

            if not model_path.exists():
                raise FileNotFoundError(f"Model {model_path} not found")
            self.config_loader.load_label.set_visibility(True)

        except Exception as e:
            ui.notify(str(e), type="negative")

    def build(self):

        self.config_loader = ConfigLoader(self.config_path)

        self.config_loader.build_config_loader()

        self.config_loader.load_btn.on_click(
            lambda: self.load_config(self.config_loader.config.value)
        )

        self.model_loader_label = ui.label("")
        self.model_loader_label.set_visibility(False)

        with ui.row().classes("w-full"):
            self.eval_btn = ui.button("Evaluate", on_click=self.start_eval).classes(
                "flex-grow"
            )

            self.stop_btn = ui.button("Stop", on_click=self.stop_eval).classes(
                "flex-grow"
            )

            self.stop_btn.set_visibility(False)

        @app.get("/video/frame")
        async def get_frame() -> Response:
            frame = self.controller.get_current_frame()
            if frame is None:
                return placeholder
            jpeg = await run.cpu_bound(frame_to_data_url, frame)
            return Response(jpeg, media_type="image/jpeg")

        with ui.row().classes("w-full justify-center"):
            self.output = ui.interactive_image("video/frame").classes(
                "w-[640px] h-[480px] object-contain"
            )
