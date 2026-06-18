from collections import deque
from pathlib import Path

import polars as pl
import seaborn as sb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from util.inspection_helper import load_algorithms
from util.utils import get_project_root, get_vec_env_class


class Train:
    def __init__(self):
        self.running = False
        self.algorithms = load_algorithms()
        self.model = None

    def train(self, config):
        self.running = True
        state = Train.TrainState()

        env_param = config.config.get("env_param")
        env_param["vec_env_cls"] = get_vec_env_class(env_param["vec_env_cls"])
        env = make_vec_env(**env_param)
        if config.config.get("vec_frame_stack").get("enabled"):
            env = VecFrameStack(env, config.config.get("vec_frame_stack").get("n_stack"))
        model_param = config.config.get("model_param")
        model_class = self.algorithms.get(config.config.get("algorithm"))
        if Path(get_project_root() / config.config.get("model_path")).exists():
            model = model_class(env=env, policy=model_param["policy"]).load(config.config.get("model_path"))
        else:
            model = model_class(env=env, **model_param)

        callback = Train.StreamingCallback(self, state)

        model.learn(total_timesteps=config.config.get("total_timesteps"), callback=callback)
        df = pl.DataFrame(state.episodes)
        fig = sb.lineplot(x="timesteps", y="reward", data=df).get_figure()
        yield "Finished training.", fig

    def stop(self):
        self.running = False

    class TrainState:
        def __init__(self):
            self.logs = deque(maxlen=1000)
            self.episodes = []

    class StreamingCallback(BaseCallback):
        def __init__(self, trainer, state, verbose=0):
            super().__init__(verbose)
            self.trainer = trainer
            self.state = state

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode" in info:
                    log = {
                        "timesteps": self.num_timesteps,
                        "reward": info["episode"]["r"],
                        "length": info["episode"]["l"],
                    }
                    self.state.logs.append(log)
                    self.state.episodes.append(log)
            return self.trainer.running
