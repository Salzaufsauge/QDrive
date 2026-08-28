from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from util.utils import copy_vecnorm, get_project_root, log
from util.video import record_and_upload_video
from util.wandb_logging import log_wandb_metrics


class MilestoneCallback(BaseCallback):
    def __init__(
        self, trainer, eval_env, milestones: list, verbose=0, n_eval_episodes=10
    ):
        super().__init__(verbose)
        self.trainer = trainer
        self.eval_env = eval_env
        self.milestones = sorted([int(milestone) for milestone in milestones])
        self.current_milestone = self.milestones.pop(0) if self.milestones else None
        self.train_start_timesteps = trainer.config.config.get("current_timesteps", 0)
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if not self.milestones and self.current_milestone is None:
            return True
        if self.train_start_timesteps + self.num_timesteps >= self.current_milestone:
            try:
                log("INFO", f"Starting milestone {self.current_milestone}")
                model_base_path = self.trainer.config.model_path.replace(".zip", "")
                video_base_path = model_base_path.replace("models", "experiments")
                copy_vecnorm(self.model, self.eval_env)
                mean_reward, std_reward = evaluate_policy(
                    self.model, self.eval_env, self.n_eval_episodes
                )
                log_wandb_metrics(
                    self.trainer.run,
                    self.current_milestone,
                    {
                        "milestone/mean_reward": mean_reward,
                        "milestone/std_reward": std_reward,
                        "milestone/n_episodes": self.n_eval_episodes,
                    },
                )
                log_wandb_metrics(
                    self.trainer.run,
                    self.current_milestone,
                    {
                        "milestone/mean_reward": mean_reward,
                        "milestone/std_reward": std_reward,
                        "milestone/n_episodes": 10,
                    },
                )
                video_path = (
                    get_project_root() / video_base_path / str(self.current_milestone)
                )
                model_path = (
                    get_project_root()
                    / model_base_path
                    / str(self.current_milestone)
                    / f"model-{mean_reward}.zip"
                )
                vecnorm_path = str(model_path).replace(".zip", ".pkl")
                vecnorm = self.model.get_vec_normalize_env()
                self.model.save(model_path)
                record_and_upload_video(
                    self.trainer,
                    self.model,
                    self.eval_env,
                    video_path,
                    caption=f"{self.trainer.config.algorithm} at step {self.current_milestone}",
                    step=self.current_milestone,
                )

                if vecnorm is not None:
                    vecnorm.save(vecnorm_path)
                    self.trainer.run.log_model(vecnorm_path)
                self.trainer.run.log_model(model_path)

                log(
                    "INFO",
                    f"Milestone {self.current_milestone} done | reward={mean_reward:.2f}",
                )

            except Exception as e:
                log("ERROR", f"Milestone evaluation failed: {e}")
                raise
            finally:
                try:
                    self.current_milestone = self.milestones.pop(0)
                    self.trainer.config.milestones = self.milestones
                except IndexError:
                    self.current_milestone = None

        return True
