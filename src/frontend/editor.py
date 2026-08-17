from pathlib import Path

from nicegui import ui

from backend.controller import Controller
from frontend.components.eval_tab import EvalTab
from frontend.components.training_tab import TrainingTab
from util.LoggingBroker import LoggingBroker


class Editor:
    def __init__(
        self, controller: Controller, logging_broker: LoggingBroker, config_path: Path
    ):
        self.config_path = config_path
        self.controller = controller
        self.logging_broker = logging_broker

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
        @ui.page("/")
        def index():
            self.build()

        ui.run(
            title="QDrive Training UI",
            host="0.0.0.0",
            reload=False,
            dark=True,
        )
