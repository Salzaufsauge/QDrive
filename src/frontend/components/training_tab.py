import asyncio
from functools import partial
from pathlib import Path

from nicegui import app, ui

from backend.config.builder import ConfigBuilder
from backend.config.storage import load_config
from backend.controller import Controller
from frontend.components.config_loader import ConfigLoader
from frontend.components.env_tab import EnvTab
from frontend.components.model_tab import ModelTab
from frontend.components.wrapper_tab import WrapperTab
from util.teestream import StreamType


class TrainingTab:
    def __init__(self, controller: Controller, config_path: Path, logging_broker):
        self.config_loader = None
        self.controller = controller
        self.config_path = config_path
        self.config = None

        self.env_tab = EnvTab()
        self.wrapper_tab = WrapperTab()
        self.model_tab = ModelTab()

        self.train_btn = None
        self.stop_btn = None

        self.graph = None
        self.figure = None
        self.training_timer = None

        self.logging_broker = logging_broker

        self.model_container = None

    def setup_config(self, *params):
        try:
            self.config = ConfigBuilder.write_config(list(params))
        except Exception as e:
            ui.notify(f"Error: {e}", type="negative")
            raise

    def get_training_state(self):
        if not self.controller.is_alive_and_training():
            self.training_timer.cancel()
        new_data = self.controller.get_training_state()
        if new_data:
            self.extend_plot(new_data)

    def reset_plot(self, num_envs):
        self.figure = {
            "data": [],
        }

        for env_idx in range(num_envs):
            self.figure["data"].append(
                {"name": f"Env {env_idx}", "type": "scatter", "x": [], "y": []}
            )

        self.graph.update_figure(self.figure)

    def extend_plot(self, data):
        for episode in data:
            env = episode["env_num"]

            x = episode["timesteps"]
            y = episode["reward"]

            self.figure["data"][env]["x"].append(x)
            self.figure["data"][env]["y"].append(y)
            self.graph.run_plot_method("extendTraces", {"x": [[x]], "y": [[y]]}, [env])

    async def stop_training(self):
        await self.controller.stop_training()

        self.stop_btn.set_visibility(False)
        self.train_btn.set_visibility(True)

    def load_config(self, config_path):
        if config_path is None and self.config is not None:
            self.config = None
            ui.notify("Unloaded config")
            return
        try:
            self.config = load_config(config_path)
            self.config_loader.load_label.set_visibility(True)
        except Exception as e:
            ui.notify(f"Error: {e}", type="negative")
            return

    def build(self):
        self.config_loader = ConfigLoader(self.config_path)
        self.config_loader.build_config_loader()
        self.config_loader.load_btn.on_click(
            lambda: self.load_config(self.config_loader.config.value)
        )

        with ui.tabs().classes("w-full justify-center") as tabs:
            env_tab_btn = ui.tab("Environment Parameters")
            wrapper_tab_btn = ui.tab("Wrapper Parameters")
            model_tab_btn = ui.tab("Model Parameters")

        with ui.tab_panels(tabs, value=env_tab_btn).classes("w-full flex-grow"):
            with ui.tab_panel(env_tab_btn):
                self.env_tab.build()

            with ui.tab_panel(wrapper_tab_btn):
                self.wrapper_tab.build()

            with ui.tab_panel(model_tab_btn):
                self.model_tab.build()

        self.train_btn = ui.button("Train", on_click=self.train).classes("w-full")

        self.stop_btn = ui.button("Stop", on_click=self.stop_training).classes("w-full")

        self.stop_btn.set_visibility(False)

        with ui.expansion("Training Output", value=True).classes("w-full"):
            with ui.row().classes("w-full"):
                with ui.keep_alive():
                    console = ui.log().classes("flex-1").style("height: 400px")
                    self.graph = ui.plotly({}).classes("flex-1").style("height: 400px")

        subscription = self.logging_broker.subscribe()
        app.on_delete(partial(self.logging_broker.unsubscribe, subscription))

        async def consume_log():
            try:
                while True:
                    msg = await subscription.client_queue.get()
                    match msg.stream_type:
                        case StreamType.STDERR:
                            console.push(msg.message, classes="text-red")
                        case StreamType.STDOUT:
                            console.push(msg.message)
            finally:
                self.logging_broker.unsubscribe(subscription)

        asyncio.create_task(consume_log())

    def train(self):
        try:
            self.setup_config(
                *[self.config_loader.config],
                *self.env_tab.env_params,
                *self.wrapper_tab.wrapper_params,
                *self.model_tab.model_params,
            )

        except Exception as e:
            raise e

        self.reset_plot(self.config.env_params["n_envs"])

        self.controller.start_training(self.config)
        self.training_timer = ui.timer(10, self.get_training_state)

        self.train_btn.set_visibility(False)
        self.stop_btn.set_visibility(True)
