import copy
import datetime
import threading
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sb
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder

from backend.configuration import Configuration
from util.inspection_helper import load_algorithms
from util.utils import get_project_root, get_vec_env_class


class Train:
    def __init__(self):
        self.running = threading.Event()
        self.algorithms = load_algorithms()
        self.config = None
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
                model = model_class.load(env=env, path=get_project_root() / config.config.get("model_path"))
            else:
                model = model_class(env=env, **model_param)

            eval_env = make_vec_env(**env_param)
            if config.config.get("vec_frame_stack").get("enabled"):
                eval_env = VecFrameStack(eval_env, config.config.get("vec_frame_stack").get("n_stack"))

            eval_env.set_attr("n_envs", 1)

            streaming_callback = StreamingCallback(self, self.state)
            milestone_callback = MilestoneCallback(self, eval_env, config.config.get("milestones"))
            callback = CallbackList([streaming_callback, milestone_callback])

            model.learn(total_timesteps=config.config.get("total_timesteps"), callback=callback)
            self.state.log("INFO", "Training finished")
        except Exception as e:
            self.state.log("ERROR", f"Training failed: {e}")
        finally:
            env.close()
            self.running.clear()

    def stop(self):
        if self.state is not None:
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
        plt.close(fig)
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
                    self.trainer.config.save_model(self.model)
        return self.trainer.running.is_set()


class MilestoneCallback(BaseCallback):
    def __init__(self, trainer, eval_env, milestones: list, verbose=0):
        super().__init__(verbose)
        self.trainer = trainer
        self.eval_env = eval_env
        self.milestones = milestones
        self.current_milestone = self.milestones.pop(0) if self.milestones else None

    def _on_step(self) -> bool:
        if not self.milestones and self.current_milestone is None:
            return True
        if self.num_timesteps >= self.current_milestone:
            path = Path(self.trainer.config.config.get("model_path").replace(".zip", ""))
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=10)
            video_path = get_project_root() / path / str(self.current_milestone)
            model_path = path / str(self.current_milestone) / f"model-{mean_reward}.zip"
            conf = copy.deepcopy(self.trainer.config.config)
            conf["model_path"] = str(model_path)
            conf = Configuration(conf)
            conf.save_model(self.model)
            rec_env = VecVideoRecorder(self.eval_env, str(video_path), record_video_trigger=lambda x: x == 0,
                                       video_length=1000)
            obs = rec_env.reset()
            for _ in range(1000):
                action, _states = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, info = rec_env.step(action)
                rec_env.render("rgb_array")
            rec_env.close()
            try:
                self.current_milestone = self.milestones.pop(0)
            except IndexError:
                self.current_milestone = None

        return True
