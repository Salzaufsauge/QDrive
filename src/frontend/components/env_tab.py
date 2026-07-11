import inspect

from nicegui import ui
from stable_baselines3.common.env_util import make_vec_env

from util.inspection_helper import make_ui_for_param
from util.utils import build_ui_params, get_envs


def make_env_ui(param: inspect.Parameter):
    if param.name == "env_id":
        return ui.select(
            options=get_envs(),
            value="CarRacing-v3",
            label=param.name
        )

    if param.name == "vec_env_cls":
        return ui.select(
            options=[
                "DummyVecEnv",
                "SubprocVecEnv"
            ],
            value="DummyVecEnv",
            label=param.name
        )

    if param.name == "wrapper_class":
        elem = ui.label("")
        elem.set_visibility(False)
        return elem

    return make_ui_for_param(param)


class EnvTab:
    def __init__(self):
        self.env_params = []

    def build(self):
        params = list(
            inspect.signature(
                make_vec_env
            ).parameters.values()
        )

        self.env_params.extend(
            build_ui_params(
                params,
                4,
                make_env_ui
            )
        )

    def get_env_params(self):
        return self.env_params