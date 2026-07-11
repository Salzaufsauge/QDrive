from nicegui import ui


class TagComponent:
    def __init__(self, value=None):
        self.value = value or []
        self.container = None
        self.input = None

        self.build()

    def build(self):
        with ui.column().classes('w-full') as self.container:
            self.render_tags()

    def render_tags(self):
        self.container.clear()

        with self.container:
            with ui.row().classes(
                    "w-full gap-2 items-center"
            ):
                for index, tag in enumerate(self.value):
                    with ui.row().classes(
                            "inline-flex items-center rounded-full bg-light-blue px-3 py-1 text-sm font-medium text-white"
                    ):
                        ui.label(str(tag))

                        ui.button(
                            icon="close",
                            on_click=lambda i=index: self.remove_tag(i)
                        ).props(
                            "flat dense round"
                        )

                self.input = ui.input(
                    placeholder="Enter a Milestone and press Enter..."
                ).classes('flex-grow')

                self.input.on(
                    "keydown.enter",
                    self.add_tag
                )

    def add_tag(self):
        text = self.input.value.strip()

        if not text:
            return

        try:
            value = int(text)
        except ValueError:
            return

        self.value.append(value)

        self.input.value = ""
        self.render_tags()

    def remove_tag(self, index):
        self.value.pop(index)
        self.render_tags()

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value or []
        self.render_tags()
