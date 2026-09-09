import numpy as np
from gymnasium import ObservationWrapper, spaces


class TMRLFullObsWrapper(ObservationWrapper):
    def __init__(
        self, env, use_only_images: bool = False, longitudinal_axis: bool = False
    ):
        """
        Initializes the observation and action spaces for a custom environment wrapper that processes
        sensor data, images, and other vehicle-related observations. Configures metadata for rendering
        and adjusts the action space if the longitudinal axis control is enabled.

        :param env: The environment being wrapped. It should have an `observation_space` of type
                    `gym.spaces.Tuple` and support the expected structure for sensor and image-based
                    data.
        :type env: gym.Env
        :param use_only_images: If True, the observation space will be reduced to only image-related
                                data (`image_history` property of the wrapped environment). Default
                                is False.
        :type use_only_images: bool
        :param longitudinal_axis: Whether to enable simplified control along the longitudinal axis.
                                  This modifies the action space to a 2-dimensional box. Default is False.
        :type longitudinal_axis: bool

        :raises TypeError: Raises an error if the provided environment's observation space is not
                           of type `gym.spaces.Tuple`.

        :attribute observation_space: Defines the structure of the processed observations returned by
                                      the wrapper. Contains elements such as `speed`, `gear`, `rpm`,
                                      `image_history`, and `action_history`.
        :type observation_space: gym.spaces.Dict | gym.spaces.Box
        :attribute action_space: Specifies the action space for the wrapped environment. If
                                 `longitudinal_axis` is enabled, it contains a 2-dimensional box
                                 space. By default, it is inherited from the environment.
        :type action_space: gym.spaces.Box
        :attribute metadata: A dictionary containing metadata for the environment, including
                             rendering modes and the calculated rendering FPS based on the parent
                             environment's time step duration.
        :type metadata: dict
        """
        super().__init__(env)

        self.use_only_images = use_only_images

        raw_space = env.observation_space
        if not isinstance(raw_space, spaces.Tuple):
            raise TypeError(
                f"{type(self).__name__} expects the the full observation space"
            )

        raw_space = raw_space.spaces

        action_dim = sum(int(space.shape[0]) for space in raw_space[4:])

        image_history = spaces.Box(
            low=0, high=255, shape=raw_space[3].shape, dtype=np.uint8
        )

        if self.use_only_images:
            self.observation_space = image_history
        self.observation_space = spaces.Dict(
            {
                "speed": raw_space[0],
                "gear": raw_space[1],
                "rpm": raw_space[2],
                "image_history": image_history,
                "action_history": spaces.Box(
                    low=-1, high=1, shape=(action_dim,), dtype=np.float32
                ),
            }
        )

        self.last_frame = None
        self.metadata = {
            **env.metadata,
            "render_modes": ["rgb_array"],
            "render_fps": round(
                1 / (env.unwrapped.time_step_duration * getattr(env, "_skip", 1))
            ),
        }

        self.longitudinal_axis = longitudinal_axis
        if longitudinal_axis:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32
            )

    @property
    def render_mode(self):
        return "rgb_array"

    def observation(self, observation):
        speed, gear, rpm, image_history, *action_history = observation
        image_history = np.asarray(image_history, dtype=np.uint8)
        self.last_frame = image_history[-1]

        if self.use_only_images:
            return image_history

        return {
            "image_history": image_history,
            "speed": speed,
            "gear": gear,
            "rpm": rpm,
            "action_history": np.concatenate(action_history).astype(np.float32),
        }

    def map_action(self, action):
        if not self.longitudinal_axis:
            return action
        lon, steer = float(action[0]), float(action[1])
        return np.array([max(lon, 0.0), max(-lon, 0.0), steer], dtype=np.float32)

    def step(self, action):
        return super().step(self.map_action(action))

    def render(self):
        if self.last_frame is None:
            return None
        frame = np.repeat(self.last_frame[:, :, np.newaxis], 3, axis=2)
        return frame.repeat(4, axis=0).repeat(4, axis=1)
