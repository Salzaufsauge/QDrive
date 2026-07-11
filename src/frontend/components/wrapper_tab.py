import inspect

from nicegui import ui

from util.inspection_helper import load_env_wrappers, make_ui_for_param
from util.utils import build_ui_params


def make_wrapper_ui(param: inspect.Parameter):
    if param.name in ["env", "venv"]:
        elem = ui.label("skip")
        elem.set_visibility(False)
        return elem

    return make_ui_for_param(param)

class WrapperTab:
    def __init__(self):
        self.wrappers = load_env_wrappers()

        self.wrapper_items = []

        self.wrapper_container = None

    def build(self):
        self.wrapper_items = []

        with ui.column().classes('w-full') as self.wrapper_container:
            for wrapper_name, wrapper_cls in self.wrappers.items():
                self.add_wrapper(wrapper_name, wrapper_cls)

    def init_sortable(self):
        self.wrapper_container.make_sortable(
            on_end=self.drag_finished
        )

    @property
    def wrapper_params(self):
        result = []

        for item in self.wrapper_items:
            result.append(item["checkbox"])

            if item["active"]:
                result.extend(item["params"])

        return result

    def add_wrapper(self, wrapper_name, wrapper_cls):
        item = {
            "name": wrapper_name,
            "active": False,
            "params": [],
            "element": None,
        }

        with ui.expansion(text=wrapper_name).classes("w-full") as card:
            item["element"] = card
            cb = ui.checkbox(
                "Enable " + wrapper_name,
                value=False,
                on_change=lambda e, i=item: self.set_active(i, e.value)
            )
            item["checkbox"] = cb
            params = list(
                inspect.signature(wrapper_cls).parameters.values()
            )
            item["params"] = build_ui_params(
                params,
                4,
                make_wrapper_ui
            )

        self.wrapper_items.append(item)

    def set_active(self, item, active):
        item["active"] = active

        self.sort_wrappers()
        self.update_ui_order()

    def sort_wrappers(self):
        active = [
            x for x in self.wrapper_items
            if x["active"]
        ]

        inactive = [
            x for x in self.wrapper_items
            if not x["active"]
        ]

        self.wrapper_items = active + inactive

    def update_ui_order(self):
        with self.wrapper_container:
            for item in self.wrapper_items:
                item["element"].move()

    def drag_finished(self, e):
        item = self.wrapper_items.pop(e.old_index)
        self.wrapper_items.insert(e.new_index, item)

        ui.notify(
            f"{item['name']} moved"
        )
