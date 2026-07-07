import copy
import threading
import time

import wandb
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from backend.callbacks import MilestoneCallback
from backend.callbacks import StreamingCallback
from backend.config.config import ExperimentConfig
from backend.state.train_state import TrainState
from util.inspection_helper import load_algorithms
from util.utils import get_project_root, get_vec_env_class


class Train:
    def __init__(self):
        self.running = threading.Event()
        self.algorithms = load_algorithms()
        self.config = None
        self.state = None
        self.run = None

    def train(self, config: ExperimentConfig):
        self.running.set()
        self.state = TrainState()
        self.config = copy.deepcopy(config)

        self.state.log("INFO", "Starting training")
        self.run = wandb.init(
            project="QDrive",
            config=config.config,
            name=config.model_path.replace(".zip", ""),
            dir=get_project_root(),
            sync_tensorboard=True,
            monitor_gym=True,
        )

        env_param = config.env_params
        env_param["vec_env_cls"] = get_vec_env_class(env_param["vec_env_cls"])
        env = make_vec_env(**env_param)
        if config.vec_frame_stack.get("enabled"):
            env = VecFrameStack(env, config.vec_frame_stack.get("n_stack"))

        try:
            model_param = config.model_params
            model_class = self.algorithms.get(config.algorithm)
            if config.abs_model_path.exists():
                model = model_class.load(env=env, path=config.abs_model_path)
            else:
                model = model_class(env=env, tensorboard_log=get_project_root() / "logs", **model_param)

            eval_env_param = copy.deepcopy(env_param)
            eval_env_param["n_envs"] = 1
            eval_env_param["vec_env_cls"] = DummyVecEnv  # for safety
            eval_env = make_vec_env(**eval_env_param)

            if config.vec_frame_stack.get("enabled"):
                eval_env = VecFrameStack(eval_env, config.vec_frame_stack.get("n_stack"))

            streaming_callback = StreamingCallback(self, self.state)
            milestone_callback = MilestoneCallback(self, eval_env, config.milestones)
            wandb_callback = WandbCallback(
                gradient_save_freq=1000,
                verbose=2,
            )
            callback = CallbackList([streaming_callback, milestone_callback, wandb_callback])

            model.learn(total_timesteps=config.config.get("total_timesteps"), tb_log_name=self.run.id,
                        callback=callback)

            self.state.log("INFO", "Training finished")

            artifact = wandb.Artifact(f"run-{self.run.id}-config", type="config")
            artifact.add_file(
                get_project_root() / config.model_path.replace("models", "experiments").replace(
                    ".zip", ".yaml"))
            self.run.log_artifact(artifact)
            self.run.log_model(config.abs_model_path, )
            self.run.finish(0)
        except Exception as e:
            if self.run is not None:
                self.run.finish(1)
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