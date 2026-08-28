import threading

import numpy as np

from backend.config.config import ExperimentConfig
from backend.env.env_manager import EnvMode, build_env
from util.inspection_helper import load_algorithms
from util.utils import load_vecnorm_stats


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
        load_vecnorm_stats(
            str(config.abs_model_path).replace(".zip", ".pkl"),
            env,
        )
        obs = env.reset()

        lstm_states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)

        try:
            while self.running.is_set():
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                obs, _rewards, done, _info = env.step(action)
                episode_starts = done
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
