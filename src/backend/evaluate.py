import threading

from backend.config.config import ExperimentConfig
from backend.env.env_manager import EnvMode, build_env
from util.inspection_helper import load_algorithms


class Evaluate:
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.running = threading.Event()
        self.algorithms = load_algorithms()
        self.current_frame = None

    def evaluate(self, config: ExperimentConfig, mode: str):
        self.running.set()

        env = build_env(config, EnvMode.EVAL)

        model = self.algorithms.get(config.algorithm).load(
            config.abs_model_path, env=env
        )
        vecnorm = model.get_vec_normalize_env()
        if vecnorm is not None:
            loaded = vecnorm.load(str(config.abs_model_path).replace(".zip", ".pkl"), venv=vecnorm)
            vecnorm.obs_rms = loaded.obs_rms
            vecnorm.ret_rms = loaded.ret_rms
        obs = env.reset()

        try:
            while self.running.is_set():
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, done, info = env.step(action)
                frame = env.render(mode=mode)
                if frame is not None:
                    with self.frame_lock:
                        self.current_frame = frame

                if done.any():
                    obs = env.reset()
        finally:
            self.running.clear()
            env.close()

    def stop(self):
        self.running.clear()

    def get_current_frame(self):
        with self.frame_lock:
            if self.current_frame is None:
                return None
            return self.current_frame
