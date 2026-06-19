import copy
import datetime
import threading
import time
from collections import deque

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from util.inspection_helper import load_algorithms
from util.utils import get_project_root, get_vec_env_class


class Train:
    def __init__(self):
        self.running = threading.Event()
        self.algorithms = load_algorithms()
        self.config = None
        self.model = None
        self.state = None

    def train(self, config):
        self.running.set()
        self.state = TrainState()
        self.config = copy.deepcopy(config)

        self.state.log("INFO", "Starting training")
        env_param = config.config.get("env_param")
        env_param["vec_env_cls"] = get_vec_env_class(env_param["vec_env_cls"])
        env = make_vec_env(**env_param)
        if config.config.get("vec_frame_stack").get("enabled"):
            env = VecFrameStack(env, config.config.get("vec_frame_stack").get("n_stack"))

        try:
            model_param = config.config.get("model_param")
            model_class = self.algorithms.get(config.config.get("algorithm"))
            if (get_project_root() / config.config.get("model_path")).exists():
                self.model = model_class.load(env=env, path=get_project_root() / config.config.get("model_path"))
            else:
                self.model = model_class(env=env, **model_param)

            callback = StreamingCallback(self, self.state)

            self.model.learn(total_timesteps=config.config.get("total_timesteps"), callback=callback)
            self.state.log("INFO", "Training finished")
        except Exception as e:
            self.state.log("ERROR", f"Training failed: {e}")
        finally:
            env.close()
            self.running.clear()

    def stop(self):
        self.state.log("INFO", "Stopping training")
        self.running.clear()

    def get_state(self):
        self.running.wait()
        while self.running.is_set():
            yield [
                "\n".join(self.state.get_logs()),
                self.state.reward_fig()
            ]
            time.sleep(10.0)
        yield [
            "\n".join(self.state.get_logs()),
            self.state.reward_fig()
        ]


class TrainState:
    def __init__(self):
        self.logs = deque(maxlen=1000)
        self.episodes = deque(maxlen=10_000)
        self.lock = threading.Lock()

    def log(self, level: str, message: str):
        with self.lock:
            self.logs.append({
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "level": level,
                "message": message,
            })

    def reward_fig(self):
        fig, ax = plt.subplots()
        with self.lock:
            if self.episodes:
                df = pl.DataFrame(self.episodes)
                sb.lineplot(
                    data=df,
                    x="timesteps",
                    y="reward",
                    ax=ax,
                )
        return fig

    def get_logs(self):
        with self.lock:
            return [f"{log['time']} - {log['level']} - {log['message']} " for log in self.logs]


class StreamingCallback(BaseCallback):
    def __init__(self, trainer, state, verbose=0):
        super().__init__(verbose)
        self.trainer = trainer
        self.state = state
        self.best_reward = -float("inf")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                log = {
                    "timesteps": self.num_timesteps,
                    "reward": info["episode"]["r"],
                    "length": info["episode"]["l"],
                }
                self.state.log("INFO",
                               f"Episode finished after {log['timesteps']} timesteps with reward {log['reward']}.")
                with self.state.lock:
                    self.state.episodes.append(log)
                if self.best_reward < log["reward"]:
                    self.state.log("INFO", f"New best reward: {log['reward']}")
                    self.best_reward = max(self.best_reward, log["reward"])
                    self.trainer.config.save_model(self.trainer.model)
        return self.trainer.running.is_set()
