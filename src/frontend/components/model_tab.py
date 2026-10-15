import inspect
from functools import partial

from nicegui import events, ui

from frontend.uituils import build_ui_params, make_ui_for_param, unwrap_ui_elem
from util.inspection_helper import (
    get_policies_from_algo,
    load_algorithms,
)


def make_model_ui(param: inspect.Parameter, algorithms, algo):
    if param.name == "policy":
        policy = list(get_policies_from_algo(algorithms[algo]).keys())

        return ui.select(value=policy[0], options=policy, label=param.name).classes(
            "flex-grow"
        )

    if param.name == "action_noise":
        with ui.row(align_items="center", wrap=False).classes("flex-grow"):
            select = ui.select(
                options={
                    None: "action noise (unknown)",
                    "NormalActionNoise": "NormalActionNoise",
                    "OrnsteinUhlenbeckActionNoise": "OrnsteinUhlenbeckActionNoise",
                },
                value=None,
                label=param.name,
                clearable=True,
            ).classes("flex-grow")
            sigma = ui.number(
                label="sigma", value=0.1, min=0, step=0.05, precision=2
            ).classes("w-24")
            sigma.bind_visibility_from(select, "value", backward=bool)
            theta = ui.number(
                label="theta", value=0.15, min=0, step=0.05, precision=2
            ).classes("w-24")
            theta.bind_visibility_from(
                select, "value", backward=lambda v: v == "OrnsteinUhlenbeckActionNoise"
            )

        select.noise_sigma = sigma
        select.noise_theta = theta
        return select

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
        self.deterministic = None
        self.n_eval_episodes = None
        self.eval_freq = None
        self.save_replay_buffer = None
        self.total_timesteps = None
        self.milestones_input = None
        self.algorithm_select = None
        self.algorithms = load_algorithms()
        self.model_params = {}
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

        self.milestones_input = ui.input_chips(
            label="Milestones",
            on_change=split_values,
            new_value_mode="add-unique",
            clearable=True,
        ).classes("milestone-chip w-full")

        self.total_timesteps = ui.number(
            label="total_timesteps", value=1000000
        ).classes("w-full")

        self.save_replay_buffer = ui.checkbox(
            text="save_replay_buffer", value=True
        ).classes("w-full")

        with (
            ui.expansion("Callback Parameters").classes("w-full"),
            ui.row().classes("w-full"),
        ):
            self.eval_freq = ui.number(label="eval_freq", value=10000).classes(
                "flex-grow"
            )

            self.n_eval_episodes = ui.number(label="n_eval_episodes", value=10).classes(
                "flex-grow"
            )

            self.deterministic = ui.checkbox(text="deterministic", value=True).classes(
                "flex-grow"
            )

        self.model_container = ui.column().classes("w-full")

        self.update_model_params(self.algorithm_select)

        self.algorithm_select.on_value_change(self.update_model_params)

    def update_model_params(self, e):
        self.model_container.clear()

        with self.model_container:
            self.get_model_params(e.value)

    def get_model_params(self, algo):
        params = list(inspect.signature(self.algorithms[algo]).parameters.values())

        self.model_params = build_ui_params(
            params, 4, partial(make_model_ui, algorithms=self.algorithms, algo=algo)
        )

    def values(self):
        return {
            "algorithm": self.algorithm_select.value,
            "milestones": self.milestones_input.value,
            "total_timesteps": int(self.total_timesteps.value),
            "save_replay_buffer": self.save_replay_buffer.value,
            "callback_params": {
                "eval_freq": int(self.eval_freq.value),
                "n_eval_episodes": int(self.n_eval_episodes.value),
                "deterministic": self.deterministic.value,
            },
            "model_param": {n: unwrap_ui_elem(e) for n, e in self.model_params.items()},
        }
