import inspect

from nicegui import ui

from util.inspection_helper import load_env_wrappers, make_ui_for_param
from util.utils import build_ui_params


def toggle_options(elems, show):
    for elem in elems:
        elem.set_visibility(show)


def make_wrapper_ui(param: inspect.Parameter):
    if param.name in ["env", "venv"]:
        elem = ui.label("skip")
        elem.set_visibility(False)
        return elem

    return make_ui_for_param(param, visible=False)

class WrapperTab:
    def __init__(self):
        self.wrappers = load_env_wrappers()
        self.wrapper_params = []

    def build(self):
        with ui.column().classes('w-full h-full'):
            for wrapper_name, wrapper_cls in self.wrappers.items():
                self.build_wrappers(wrapper_name, wrapper_cls)

    def build_wrappers(self, wrapper_name, wrapper_cls):
        with ui.card().classes('w-full h-full'):
            params = list(
                inspect.signature(wrapper_cls).parameters.values()
            )

            cb = ui.checkbox(
                wrapper_name,
                value=False
            )

            self.wrapper_params.append(cb)

            temp = build_ui_params(
                params,
                4,
                make_wrapper_ui
            )

            self.wrapper_params.extend(temp)

            def on_toggle(e):
                for elem in temp:
                    if elem is not None and getattr(elem, "text", "") != "skip":
                        elem.set_visibility(e.value)

            cb.on_value_change(on_toggle)
