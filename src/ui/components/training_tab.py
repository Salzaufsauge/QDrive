import inspect
from pathlib import Path

import gradio as gr
import seaborn as sb
from stable_baselines3.common.env_util import make_vec_env

from ui.components.model_loader import ModelLoader
from util.gym_helper import get_envs
from util.inspection_helper import make_ui_for_param, get_policies_from_algo, load_algorithms


def toggle_options(show):
    return gr.update(visible=show)


class TrainingTab:
    def __init__(self, model_path: Path | str = ".", ):
        self.model_path = model_path
        self.running = False
        self.algorithms = load_algorithms()

    def start_training(self, *params):
        self.running = True
        log = "Starting training..."

        plot = sb.lineplot(x=[], y=[], markers=True)

        try:
            while self.running:
                yield [
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(value=log),
                    gr.update(value=plot.get_figure())
                ]
        except Exception as e:
            log += f"Training failed: {e}"
            yield [
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(value=log),
                gr.update(value=plot.get_figure())
            ]
        finally:
            log += "Training stopped."
            yield [
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(value=log),
                gr.update(value=plot.get_figure())
            ]

    def stop_training(self):
        self.running = False
        yield [
            gr.update(visible=False),
            gr.update(visible=True),
        ]

    def build(self):
        with gr.Tab("Train"):
            env_params = []
            model_params = []

            with gr.Group():

                ModelLoader(self.model_path).build_model_loader()

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
                                    continue
                                env_params.append(make_ui_for_param(param))

                    with gr.Row():
                        cb = gr.Checkbox(label="Use FrameStack", value=False)
                        frame_stack = gr.Number(label="FrameStack Size", value=4, visible=False)
                        env_params.append(frame_stack)
                        cb.change(toggle_options, cb, [frame_stack])

                with gr.Accordion("Model Parameters", open=False):
                    model_params.append(gr.Dropdown(value="PPO", choices=self.algorithms.keys(), label="Algorithm"))

                    @gr.render(inputs=model_params[0])
                    def get_model_params(algo):
                        temp = list()
                        params = list(inspect.signature(self.algorithms[algo]).parameters.values())

                        for i in range(0, len(params), 3):
                            with gr.Row():
                                for param in params[i:i + 3]:
                                    if param.name == "policy":
                                        temp.append(gr.Dropdown(value="MlpPolicy", choices=get_policies_from_algo(
                                            self.algorithms[algo]).keys(), label=param.name))
                                        continue
                                    if param.name == "env": continue
                                    temp.append(make_ui_for_param(param))
                        temp.append(gr.Number(label="Total Timesteps", value=1000000))
                        model_params[1:] = temp

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

            train.click(self.start_training, inputs=env_params + model_params, outputs=[train, stop, console, graph])
            stop.click(self.stop_training, outputs=[stop, train])
