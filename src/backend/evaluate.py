import threading

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from backend.config.config import ExperimentConfig
from util.inspection_helper import load_algorithms
from util.utils import get_vec_env_class


class Evaluate:
    def __init__(self):
        self.running = threading.Event()
        self.algorithms = load_algorithms()

    def evaluate(self, config: ExperimentConfig, mode: str):
        self.running.set()
        env_param = dict(config.env_params or {})
        env_param["n_envs"] = 1
        env_param["vec_env_cls"] = get_vec_env_class(env_param["vec_env_cls"])
        env = make_vec_env(**env_param)
        if config.vec_frame_stack.get("enabled"):
            env = VecFrameStack(env, config.vec_frame_stack.get("n_stack"))
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
