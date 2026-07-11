from pathlib import Path

from nicegui import ui

from backend.controller import Controller
from frontend.components.eval_tab import EvalTab
from frontend.components.training_tab import TrainingTab


class Editor:
    def __init__(self, controller: Controller, config_path: Path):
        self.training_tab = TrainingTab(
            controller,
            config_path=config_path
        )

        self.eval_tab = EvalTab(
            controller,
            config_path=config_path
        )

        self.config_path = config_path

    def build(self):
        ui.query('body').classes('m-0 p-0')

        with ui.column().classes(
                'w-full items-center justify-start p-6 gap-6'
        ):
            ui.markdown("# QDrive Training UI").classes(
                'text-center w-full max-w-5xl'
            )

            with ui.tabs().classes(
                    'w-full justify-center'
            ) as tabs:
                train_tab = ui.tab("Train")
                eval_tab = ui.tab("Eval")

            with ui.tab_panels(
                    tabs,
                    value=train_tab
            ).classes(
                'w-full flex-grow'
            ):
                with ui.tab_panel(train_tab):
                    with ui.card().classes('w-full h-full'):
                        self.training_tab.build()

                with ui.tab_panel(eval_tab):
                    with ui.card().classes('w-full h-full'):
                        self.eval_tab.build()

    def launch(self):
        self.build()

        ui.run(
            title="QDrive Training UI",
            host="0.0.0.0",
            reload=False,
            dark=True,
            native=True
        )
