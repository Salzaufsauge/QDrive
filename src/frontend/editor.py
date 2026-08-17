import base64
from pathlib import Path

from fastapi import Response
from nicegui import app, run, ui

from backend.controller import Controller
from frontend.components.eval_tab import EvalTab
from frontend.components.training_tab import TrainingTab
from util.LoggingBroker import LoggingBroker
from util.utils import frame_to_data_url

black_1px = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII="
placeholder = Response(
    content=base64.b64decode(black_1px.encode("ascii")), media_type="image/png"
)


class Editor:
    def __init__(
        self, controller: Controller, logging_broker: LoggingBroker, config_path: Path
    ):
        self.config_path = config_path
        self.controller = controller 
        self.logging_broker = logging_broker

    def register_video_route(self) -> None:
        @app.get("/video/frame")
        async def get_frame() -> Response:
            frame = self.controller.get_current_frame()
            if frame is None:
                return placeholder
            jpeg = await run.cpu_bound(frame_to_data_url, frame)
            return Response(jpeg, media_type="image/jpeg")

    def build(self):
        ui.query("body").classes("m-0 p-0")

        with ui.column().classes("w-full items-center justify-start p-6 gap-6"):
            ui.markdown("# QDrive Training UI").classes("text-center w-full max-w-5xl")

            with ui.tabs().classes("w-full justify-center") as tabs:
                train_tab = ui.tab("Train")
                eval_tab = ui.tab("Eval")

            with ui.tab_panels(tabs, value=train_tab).classes("w-full flex-grow"):
                with ui.tab_panel(train_tab), ui.card().classes("w-full h-full"):
                    TrainingTab(
                        self.controller, self.config_path, self.logging_broker
                    ).build()

                with ui.tab_panel(eval_tab), ui.card().classes("w-full h-full"):
                    EvalTab(self.controller, self.config_path).build()

    def launch(self):
        self.register_video_route()

        @ui.page("/")
        def index():
            self.build()

        ui.run(
            title="QDrive Training UI",
            host="0.0.0.0",
            reload=False,
            dark=True,
        )
