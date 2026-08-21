import inspect
from functools import partial

from nicegui import events, ui

from util.inspection_helper import (
    get_policies_from_algo,
    load_algorithms,
    make_ui_for_param,
)
from util.utils import build_ui_params


def make_model_ui(param: inspect.Parameter, algorithms, algo):
    if param.name == "policy":
        policy = list(get_policies_from_algo(algorithms[algo]).keys())

        return ui.select(value=policy[0], options=policy, label=param.name).classes(
            "flex-grow"
        )

    if param.name in ("env", "tensorboard_log"):
        elem = ui.label("")
        elem.set_visibility(False)
        return elem

    return make_ui_for_param(param)


def split_values(e: events.ValueChangeEventArguments):
    values = [
        word.strip()
        for part in e.value
        for word in part.split(",")
        if word.strip().isdigit() and int(word.strip()) > 0
    ]
    e.sender.value = sorted(set(values), key=int)


class ModelTab:
    def __init__(self):
        self.algorithm_select = None
        self.algorithms = load_algorithms()
        self.model_params = []
        self.model_params_base_len = 0
        self.model_container = None

        ui.add_head_html(
            """
        <style type="text/tailwindcss">
        @layer components {
            .milestone-chip .q-chip {
                @apply bg-[#5898d4] text-white rounded-full px-3 py-1;
            }
        }
        </style>
        """,
            shared=True,
        )

    def build(self):
        self.algorithm_select = ui.select(
            options=list(self.algorithms.keys()), value="PPO", label="algorithm"
        ).classes("w-full")

        self.model_params.append(self.algorithm_select)

        self.model_params.append(
            ui.input_chips(
                label="Milestones",
                on_change=split_values,
                new_value_mode="add-unique",
                clearable=True,
            ).classes("milestone-chip w-full")
        )

        self.model_params.append(
            ui.number(label="total_timesteps", value=1000000).classes("w-full")
        )

        with (
            ui.expansion("Callback Parameters").classes("w-full"),
            ui.row().classes("w-full"),
        ):
            self.model_params.append(
                ui.number(label="eval_freq", value=10000).classes("flex-grow")
            )
            self.model_params.append(
                ui.number(label="n_eval_episodes", value=10).classes("flex-grow")
            )
            self.model_params.append(
                ui.checkbox(text="deterministic", value=True).classes("flex-grow")
            )

        self.model_params_base_len = len(self.model_params)

        self.model_container = ui.column().classes("w-full")

        self.update_model_params(self.model_params[0])

        self.model_params[0].on_value_change(self.update_model_params)

    def update_model_params(self, e):
        self.model_container.clear()

        with self.model_container:
            self.get_model_params(e.value)

    def get_model_params(self, algo):
        params = list(inspect.signature(self.algorithms[algo]).parameters.values())

        temp = build_ui_params(
            params, 4, partial(make_model_ui, algorithms=self.algorithms, algo=algo)
        )

        self.model_params[self.model_params_base_len :] = temp

        return self.model_params
