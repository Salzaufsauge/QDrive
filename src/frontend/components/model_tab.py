import inspect
from functools import partial

from nicegui import ui

from frontend.components import TagComponent
from util.inspection_helper import load_algorithms, get_policies_from_algo, make_ui_for_param
from util.utils import build_ui_params


def make_model_ui(param: inspect.Parameter, algorithms, algo):
    if param.name == "policy":
        policy = list(get_policies_from_algo(algorithms[algo]).keys())

        return ui.select(
            value=policy[0],
            options=policy,
            label=param.name
        ).classes('flex-grow')

    if param.name in ("env", "tensorboard_log"):
        elem = ui.label("")
        elem.set_visibility(False)
        return elem

    return make_ui_for_param(param)


class ModelTab:
    def __init__(self):
        self.algorithm_select = None
        self.algorithms = load_algorithms()
        self.model_params = []
        self.model_params_base_len = 0

    def build(self):
        self.algorithm_select = ui.select(
            options=list(self.algorithms.keys()),
            value="PPO",
            label="algorithm"
        ).classes('w-full')

        self.model_params.append(self.algorithm_select)

        self.model_params.append(
            TagComponent.TagComponent()
        )

        self.model_params.append(
            ui.number(
                label="total_timesteps",
                value=1000000
            ).classes('w-full')
        )

        with ui.expansion("Callback Parameters").classes('w-full'):
            with ui.row().classes('w-full'):
                self.model_params.append(
                    ui.number(
                        label="eval_freq",
                        value=10000
                    ).classes('flex-grow')
                )

                self.model_params.append(
                    ui.number(
                        label="n_eval_episodes",
                        value=10
                    ).classes('flex-grow')
                )

                self.model_params.append(
                    ui.checkbox(
                        text="deterministic",
                        value=True
                    ).classes('flex-grow')
                )

        self.model_params_base_len = len(self.model_params)

    def get_model_params(self, algo):
        params = list(
            inspect.signature(
                self.algorithms[algo]
            ).parameters.values()
        )

        temp = build_ui_params(
            params,
            4,
            partial(
                make_model_ui,
                algorithms=self.algorithms,
                algo=algo
            )
        )

        self.model_params[self.model_params_base_len:] = temp

        return self.model_params
