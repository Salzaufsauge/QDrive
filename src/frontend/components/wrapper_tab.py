import inspect

import gradio as gr

from util.inspection_helper import load_env_wrappers, make_ui_for_param


def toggle_options(show):
    return gr.update(visible=show)


class WrapperTab:
    def __init__(self):
        self.wrappers = load_env_wrappers()
        self.wrapper_params = list()

    def build(self):
        for wrapper_name, wrapper_cls in self.wrappers.items():
            self.build_wrappers(wrapper_name, wrapper_cls)

    def build_wrappers(self, wrapper_name, wrapper_cls):
        with gr.Column():
            params = list(inspect.signature(wrapper_cls).parameters.values())
            cb = gr.Checkbox(label=wrapper_name, value=False)
            self.wrapper_params.append(cb)
            temp = list()
            for i in range(0, len(params), 3):
                with gr.Row():
                    for param in params[i:i + 3]:
                        if param.name in ["env", "venv"]:
                            self.wrapper_params.append(gr.Label(value=None, visible=False))
                            continue
                        elem = make_ui_for_param(param, visible=False)
                        temp.append(elem)
                        self.wrapper_params.append(elem)
            for elem in temp:
                cb.change(toggle_options, cb, [elem])
