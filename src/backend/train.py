import copy
import threading

from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback

import wandb
from backend.callbacks import MilestoneCallback, StreamingCallback
from backend.config.config import ExperimentConfig
from backend.env.env_manager import EnvMode, build_env
from backend.state.train_state import TrainState
from util.inspection_helper import load_algorithms
from util.utils import get_project_root, log


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

        log("INFO", "Starting training")
        log("INFO", f"Training config: {config.config}")

        run_name: str = (
            config.model_path.removeprefix("models/")
            .replace(".zip", "")
            .replace("model-", "")
        )

        env = None
        eval_env = None

        try:
            env = build_env(config, EnvMode.TRAIN)
            eval_env = build_env(config, EnvMode.EVAL)

            model_param = config.model_params
            model_class = self.algorithms.get(config.algorithm)
            if config.abs_model_path.exists():
                model = model_class.load(env=env, path=config.abs_model_path)
                vecnorm = model.get_vec_normalize_env()
                if vecnorm is not None:
                    vecnorm.load(
                        str(config.abs_model_path).replace(".zip", ".pkl"), venv=vecnorm
                    )
            else:
                model = model_class(
                    **(
                        model_param
                        | dict(env=env, tensorboard_log=get_project_root() / "logs")
                    )
                )

            self.run = wandb.init(
                project="QDrive",
                entity="QDrive",
                config=config.config,
                name=run_name,
                dir=get_project_root(),
                sync_tensorboard=True,
                monitor_gym=True,
            )

            streaming_callback = StreamingCallback(
                self,
                self.state,
                eval_env,
                eval_freq=max(
                    config.callback_params["eval_freq"]
                    // config.env_params.get("n_envs"),
                    1,
                ),
                n_eval_episodes=config.callback_params["n_eval_episodes"],
                deterministic=config.callback_params["deterministic"],
            )
            milestone_callback = (
                MilestoneCallback(self, eval_env, config.milestones)
                if config.milestones
                else None
            )
            wandb_callback = WandbCallback(
                gradient_save_freq=1000,
                verbose=2,
            )
            callbacks = [streaming_callback, wandb_callback]
            if milestone_callback:
                callbacks.append(milestone_callback)
            callback = CallbackList(callbacks)

            model.learn(
                total_timesteps=config.config.get("total_timesteps"),
                tb_log_name=self.run.id,
                callback=callback,
            )

            log("INFO", "Training finished")

            artifact = wandb.Artifact(f"run-{self.run.id}-config", type="config")
            artifact.add_file(
                get_project_root()
                / config.model_path.replace("models", "experiments").replace(
                    ".zip", ".yaml"
                )
            )
            self.run.log_artifact(artifact)
            self.run.log_model(
                config.abs_model_path,
            )
            self.run.finish(0)
        except Exception as e:
            if self.run is not None:
                self.run.finish(1)
            log("ERROR", f"Training failed: {e}")
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    log("ERROR", "Failed to close training environment")
            if eval_env is not None:
                try:
                    eval_env.close()
                except Exception:
                    log("ERROR", "Failed to close evaluation environment")
            self.state = None
            self.config = None
            self.run = None
            self.running.clear()

    def stop(self):
        if self.state is not None:
            log("INFO", "Stopping training")
        self.running.clear()

    def get_state(self):
        return self.state.reward_fig()
