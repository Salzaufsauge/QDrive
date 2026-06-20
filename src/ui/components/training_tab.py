import datetime
import inspect
import threading
from pathlib import Path

import gradio as gr
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from backend.configuration import Configuration
from backend.controller import Controller
from ui.components.model_loader import ModelLoader
from util.inspection_helper import make_ui_for_param, get_policies_from_algo, load_algorithms
from util.utils import get_envs, replace_empty_strings


def toggle_options(show):
    return gr.update(visible=show)


def __build_config__(params, sig_params):
    temp = dict()
    for key in sig_params.keys():
        val = params.pop(0)
        if val is not None:
            if key.endswith("kwargs"):
                temp[key] = {row[0]: row[1] for row in val if row and row[1] not in [None, ""]}
            else:
                temp[key] = val
    return temp


class TrainingTab:
    def __init__(self, controller: Controller, model_path: Path):
        self.controller = controller
        self.model_path = model_path
        self.config = Configuration()
        self.running = False
        self.algorithms = load_algorithms()

    def setup_config(self, *params):
        conf = dict()
        params = list(params)
        model_path = params.pop(0)
        if model_path is not None:
            self.config.load_model(model_path)
            return
        conf["env_param"] = dict()
        env_params = inspect.signature(make_vec_env).parameters
        conf["env_param"] = conf["env_param"] | __build_config__(params, env_params)
        conf["vec_frame_stack"] = dict()
        conf["vec_frame_stack"]["enabled"] = params.pop(0)
        vec_frame_stack_params = inspect.signature(VecFrameStack).parameters
        conf["vec_frame_stack"] = conf["vec_frame_stack"] | __build_config__(params, vec_frame_stack_params)
        conf["model_param"] = dict()
        conf["algorithm"] = params.pop(0)
        model_params = inspect.signature(self.algorithms[conf["algorithm"]]).parameters
        conf["model_param"] = conf["model_param"] | __build_config__(params, model_params)
        conf["total_timesteps"] = params.pop(0)
        conf[
            "model_path"] = f"models/{conf['env_param']['env_id']}/{conf['algorithm']}/model-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip"
        conf = replace_empty_strings(conf)
        self.config.config = conf

    def start_training(self):
        thread = threading.Thread(
            target=self.controller.start_training,
            args=(self.config,),
            daemon=True,
        )
        thread.start()

    def get_training_state(self):
        yield from self.controller.get_training_state()

    def stop_training(self):
        self.controller.stop_training()
        yield [
            gr.update(visible=False),
            gr.update(visible=True),
        ]

    def load_model(self, model_path: Path):
        self.config.load_model(model_path)
        return gr.update(visible=True, value=f"Model {model_path} loaded")

    def build(self):
        with gr.Tab("Train"):
            env_params = []
            model_params = []

            with gr.Group():

                model_loader = ModelLoader(self.model_path)
                model_loader.build_model_loader()
                model_loader.load_btn.click(self.load_model, inputs=[model_loader.model],
                                            outputs=model_loader.load_label, )

                with gr.Accordion("Environment Parameters", open=False):
                    params = list(inspect.signature(make_vec_env).parameters.values())
                    for i in range(0, len(params), 3):
                        with gr.Row():
                            for param in params[i:i + 3]:
                                if param.name == "env_id":
                                    env_params.append(
                                        gr.Dropdown(value="CarRacing-v3", choices=get_envs(), label=param.name))
                                    continue
                                if param.name == "vec_env_cls":
                                    env_params.append(
                                        gr.Dropdown(value="DummyVecEnv", choices=["DummyVecEnv", "SubprocVecEnv"],
                                                    label=param.name))
                                    continue
                                if param.name == "wrapper_class":
                                    env_params.append(gr.Label(value=None, visible=False))
                                    continue
                                env_params.append(make_ui_for_param(param))

                    with gr.Row():
                        params = list(inspect.signature(VecFrameStack).parameters.values())
                        cb = gr.Checkbox(label="Use FrameStack", value=False)
                        env_params.append(cb)
                        for i in range(0, len(params), 3):
                            with gr.Row():
                                for param in params[i:i + 3]:
                                    if param.name == "n_stack":
                                        frame_stack = gr.Number(label=param.name, value=param.default, visible=False)
                                        env_params.append(frame_stack)
                                    else:
                                        env_params.append(gr.Label(value=None, visible=False))

                        cb.change(toggle_options, cb, [frame_stack])

                with gr.Accordion("Model Parameters", open=False):
                    model_params.append(gr.Dropdown(value="PPO", choices=self.algorithms.keys(), label="algorithm"))

                    @gr.render(inputs=model_params[0])
                    def get_model_params(algo):
                        temp = list()
                        params = list(inspect.signature(self.algorithms[algo]).parameters.values())

                        for i in range(0, len(params), 3):
                            with gr.Row():
                                for param in params[i:i + 3]:
                                    if param.name == "policy":
                                        policy = get_policies_from_algo(self.algorithms[algo]).keys()
                                        temp.append(gr.Dropdown(choices=policy, label=param.name, interactive=True))
                                        continue
                                    if param.name == "env":
                                        temp.append(gr.Label(value=None, visible=False))
                                        continue
                                    if param.name in ["learning_rate", "clip_range"]:
                                        temp.append(gr.Number(label=param.name, value=param.default, interactive=True))
                                        continue
                                    temp.append(make_ui_for_param(param))
                        temp.append(gr.Number(label="total_timesteps", value=1000000, interactive=True))
                        model_params[1:] = temp
                        train.click(self.setup_config, inputs=[model_loader.model] + env_params + model_params,
                                    ).then(lambda: (gr.update(visible=False), gr.update(visible=True)),
                                           outputs=[train, stop]
                                           ).then(self.start_training).then(self.get_training_state,
                                                                            outputs=[console, graph])

            train = gr.Button("Train")
            stop = gr.Button("Stop", visible=False)

            with gr.Accordion("Training Output", open=True):
                with gr.Row():
                    console = gr.Textbox(
                        label="Console",
                        lines=15,
                        interactive=False
                    )
                    graph = gr.Plot(label="Training Curve")

            stop.click(self.stop_training, outputs=[stop, train])
