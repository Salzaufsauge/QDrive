import numpy as np
from gymnasium import ObservationWrapper, spaces


class TMRLFullObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        raw = env.observation_space
        assert isinstance(raw, spaces.Tuple)

        raw_space = raw.spaces

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

    def observation(self, observation):
        (speed, gear, rpm, image_history, action_0, action_1) = observation

        return {
            "image_history": np.asarray(image_history, dtype=np.uint8),
            "speed": speed,
            "gear": gear,
            "rpm": rpm,
            "action_history": np.concatenate((action_0, action_1)).astype(np.float32),
        }
