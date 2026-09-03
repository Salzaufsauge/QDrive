import numpy as np
from gymnasium import ObservationWrapper, spaces


class TMRLFullObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        raw_space = env.observation_space
        if not isinstance(raw_space, spaces.Tuple):
            raise TypeError(
                f"{type(self).__name__} expects the the full observation space"
            )

        raw_space = raw_space.spaces

        action_dim = sum(int(space.shape[0]) for space in raw_space[4:])
        self.observation_space = spaces.Dict(
            {
                "speed": raw_space[0],
                "gear": raw_space[1],
                "rpm": raw_space[2],
                "image_history": spaces.Box(
                    low=0, high=255, shape=raw_space[3].shape, dtype=np.uint8
                ),
                "action_history": spaces.Box(
                    low=-1, high=1, shape=(action_dim,), dtype=np.float32
                ),
            }
        )

        self.last_frame = None
        self.metadata = {
            **env.metadata,
            "render_modes": ["rgb_array"],
            "render_fps": round(1 / env.unwrapped.time_step_duration),
        }

    @property
    def render_mode(self):
        return "rgb_array"

    def observation(self, observation):
        speed, gear, rpm, image_history, *action_history = observation
        image_history = np.asarray(image_history, dtype=np.uint8)
        self.last_frame = image_history[-1]

        return {
            "image_history": np.asarray(image_history, dtype=np.uint8),
            "speed": speed,
            "gear": gear,
            "rpm": rpm,
            "action_history": np.concatenate(action_history).astype(np.float32),
        }

    def render(self):  # mode="human" mit Leon abklären
        if self.last_frame is None:
            return None
        return np.repeat(self.last_frame[:, :, np.newaxis], 3, axis=2)
