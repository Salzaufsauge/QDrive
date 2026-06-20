from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from backend.configuration import Configuration
from util.inspection_helper import load_algorithms
from util.utils import get_vec_env_class, get_project_root


class Evaluate:
    def __init__(self):
        self.running = False
        self.algorithms = load_algorithms()

    def evaluate(self, config: Configuration):
        self.running = True
        env_param = config.config.get("env_param")
        env_param["n_envs"] = 1
        env_param["vec_env_cls"] = get_vec_env_class(env_param["vec_env_cls"])
        env = make_vec_env(**env_param)
        if config.config.get("vec_frame_stack").get("enabled"):
            env = VecFrameStack(env, config.config.get("vec_frame_stack").get("n_stack"))
        model = self.algorithms.get(config.config.get("algorithm")).load(
            get_project_root() / config.config.get("model_path"), env=env)
        obs = env.reset()
        try:
            while self.running:
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, done, info = env.step(action)
                yield env.render("rgb_array")
        finally:
            env.close()

    def stop(self):
        self.running = False
