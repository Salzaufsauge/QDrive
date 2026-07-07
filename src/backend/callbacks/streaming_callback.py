from stable_baselines3.common.callbacks import BaseCallback

from backend.config.storage import save_config, save_model


class StreamingCallback(BaseCallback):
    def __init__(self, trainer, state, verbose=0):
        super().__init__(verbose)
        self.trainer = trainer
        self.state = state
        self.best_reward = -float("inf")
        self.last_timesteps = 0

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
                    self.trainer.config.current_timesteps = self.num_timesteps - self.last_timesteps + self.trainer.config.config.get(
                        "current_timesteps", 0)
                    self.last_timesteps = self.num_timesteps
                    save_model(self.trainer.config, self.model)
                    save_config(self.trainer.config)
        return self.trainer.running.is_set()
