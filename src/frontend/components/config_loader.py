from pathlib import Path

from nicegui import ui


class ConfigLoader:
    def __init__(self, config_path: Path):
        self.load_label = None
        self.load_btn = None
        self.config = None
        self.config_path = config_path

        self.timer = None

    def refresh_configs(self):
        choices = [
            str(p.relative_to(self.config_path.parent))
            for p in self.config_path.glob("**/*.yaml")
        ]

        current = self.config.value

        self.config.options = choices

        if current in choices:
            self.config.value = current

        self.config.update()

    def build_config_loader(self):
        configs = [
            str(p.relative_to(self.config_path.parent))
            for p in self.config_path.glob("**/*.yaml")
        ]

        with ui.row().classes("w-full"):
            self.config = ui.select(
                options=configs,
                label="Config",
                value=None,
                on_change=lambda x: self.load_label.set_visibility(False),
                clearable=True,
            ).classes("flex-grow")

            with ui.column().classes("flex-grow"):
                self.load_btn = ui.button("Load Config").classes("flex-grow w-full")

                self.load_label = ui.label("Config Loaded").classes("flex-grow w-full")

                self.load_label.set_visibility(False)

        self.timer = ui.timer(5.0, self.refresh_configs)
