from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from backend.config.storage import save_config, save_model
from backend.state.train_state import log
from util.utils import copy_vecnorm


class StreamingCallback(BaseCallback):
    def __init__(
        self,
        trainer,
        state,
        eval_env,
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    ):
        super().__init__(verbose)
        self.trainer = trainer
        self.state = state
        self.best_reward = -float("inf")
        self.last_timesteps = 0
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            log("INFO", "Running evaluation...")
            copy_vecnorm(self.model, self.eval_env)
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                render=False,
            )
            log(
                "INFO",
                f"Evaluation over {self.n_eval_episodes} episodes: {mean_reward:.2f} +/- {std_reward:.2f}",
            )
            if mean_reward > self.best_reward:
                log("INFO", f"New best reward: {mean_reward:.2f}")
                self.best_reward = mean_reward
                self.trainer.config.current_timesteps = (
                    self.num_timesteps
                    - self.last_timesteps
                    + self.trainer.config.config.get("current_timesteps", 0)
                )
                self.last_timesteps = self.num_timesteps
                save_model(self.trainer.config, self.model)
                save_config(self.trainer.config)
        for env_idx, info in enumerate(infos):
            if "episode" in info:
                episode = {
                    "env_num": env_idx,
                    "timesteps": self.num_timesteps,
                    "reward": info["episode"]["r"],
                    "length": info["episode"]["l"],
                }
                log(
                    "INFO",
                    f"Episode finished after {episode['timesteps']} timesteps with reward {episode['reward']}.",
                )
                with self.state.lock:
                    self.state.episodes.append(episode)
        return self.trainer.running.is_set()
