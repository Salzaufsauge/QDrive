import inspect

import gradio as gr

from util.inspection_helper import load_env_wrappers, make_ui_for_param
from util.utils import build_ui_params


def toggle_options(show):
    return gr.update(visible=show)


def make_wrapper_ui(param: inspect.Parameter):
    if param.name in ["env", "venv"]:
        return gr.Label(label="skip", value=None, visible=False)
    else:
        return make_ui_for_param(param, visible=False)

class WrapperTab:
    def __init__(self):
        self.wrappers = load_env_wrappers()
        self.wrapper_params = list()

    def build(self):
        with gr.Draggable():
            for wrapper_name, wrapper_cls in self.wrappers.items():
                self.build_wrappers(wrapper_name, wrapper_cls)

    def build_wrappers(self, wrapper_name, wrapper_cls):
        with gr.Group():
            params = list(inspect.signature(wrapper_cls).parameters.values())
            cb = gr.Checkbox(label=wrapper_name, value=False)
            self.wrapper_params.append(cb)
            temp = build_ui_params(params, 4, make_wrapper_ui)
            self.wrapper_params.extend(temp)
            for elem in temp:
                if elem.label != "skip":
                    cb.change(toggle_options, cb, [elem])
