import copy
import threading
import traceback

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import configure_logger
from wandb.integration.sb3 import WandbCallback

import wandb
from backend.callbacks import MilestoneCallback, StreamingCallback
from backend.config.config import ExperimentConfig
from backend.env.env_manager import EnvMode, build_env
from backend.state.train_state import TrainState
from util.inspection_helper import load_algorithms
from util.utils import build_action_noise, get_project_root, load_vecnorm_stats, log
from util.video import record_pending_best_model
from util.wandb_logging import WandbOutputFormat, configure_wandb_metrics


class Train:
    def __init__(self):
        self.running = threading.Event()
        self.algorithms = load_algorithms()
        self.config = None
        self.state = None
        self.run = None
        self.pending_best_model = None

    def train(self, config: ExperimentConfig):
        self.running.set()
        self.state = TrainState()
        self.config = copy.deepcopy(config)
        train_start_timesteps = int(config.config.get("current_timesteps", 0))

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
                load_vecnorm_stats(
                    str(config.abs_model_path).replace(".zip", ".pkl"),
                    env,
                )
            else:
                model_override = {"env": env}
                if "action_noise" in model_param:
                    model_override["action_noise"] = build_action_noise(model_param["action_noise"], env)

                model = model_class(**(model_param | model_override))

            self.run = wandb.init(
                project="QDrive",
                entity="QDrive",
                config=config.config,
                name=run_name,
                dir=get_project_root(),
                monitor_gym=True,
            )

            configure_wandb_metrics(self.run)
            model_logger = configure_logger(
                model.verbose,
                reset_num_timesteps=True,
            )
            model.set_logger(model_logger)
            model_logger.output_formats.append(
                WandbOutputFormat(
                    self.run,
                    step_offset=train_start_timesteps,
                )
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
                MilestoneCallback(
                    self,
                    eval_env,
                    config.milestones,
                    n_eval_episodes=config.callback_params["n_eval_episodes"],
                )
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
                callback=callback,
            )

            log("INFO", "Training finished")

            record_pending_best_model(
                self,
                eval_env,
                history_step=train_start_timesteps
                + int(model.num_timesteps)
                + 1,  # +1 in case of last milestone == total_timesteps
            )

            artifact = wandb.Artifact(f"run-{self.run.id}-config", type="config")
            artifact.add_file(
                get_project_root()
                / config.model_path.replace("models", "experiments").replace(
                    ".zip", ".yaml"
                )
            )
            self.run.log_artifact(artifact)

            vecnorm = model.get_vec_normalize_env()
            if vecnorm is not None:
                vecnorm_path = str(config.abs_model_path).replace(".zip", ".pkl")
                vecnorm.save(vecnorm_path)
                self.run.log_model(vecnorm_path)

            self.run.log_model(config.abs_model_path)
            self.run.finish(0)
        except Exception:
            failure = traceback.format_exc()
            if self.run is not None:
                try:
                    self.run.finish(1)
                except Exception:
                    failure += "\nWandb failed to finish run:"
                    failure += traceback.format_exc()
            log("ERROR", f"Training failed: {failure}")
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

    def get_state(self, sequence_after: int = 0):
        if self.state is None:
            return []
        return self.state.reward_fig(sequence_after)
