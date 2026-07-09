import threading

from backend.config.config import ExperimentConfig
from backend.env.env_manager import EnvMode, build_env
from util.inspection_helper import load_algorithms


class Evaluate:
    def __init__(self):
        self.running = threading.Event()
        self.algorithms = load_algorithms()

    def evaluate(self, config: ExperimentConfig, mode: str):
        self.running.set()

        env = build_env(config, EnvMode.EVAL)

        model = self.algorithms.get(config.algorithm).load(
            config.abs_model_path, env=env)
        obs = env.reset()
        try:
            while self.running.is_set():
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, done, info = env.step(action)
                yield env.render(mode)
        finally:
            self.running.clear()
            env.close()

    def stop(self):
        self.running.clear()
