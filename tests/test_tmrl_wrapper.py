from typing import ClassVar

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from backend.env.wrappers import TMRLFullObsWrapper


class SyntheticTMRLEnv(gym.Env):
    time_step_duration = 0.05
    _skip = 2
    metadata: ClassVar[dict] = {"source": "synthetic"}

    def __init__(self):
        self.action_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        scalar = spaces.Box(0, 10000, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Tuple(
            (
                scalar,
                scalar,
                scalar,
                spaces.Box(0, 255, shape=(2, 2, 3), dtype=np.uint8),
                self.action_space,
                self.action_space,
            )
        )
        self.raw = (
            np.array([10], dtype=np.float32),
            np.array([2], dtype=np.float32),
            np.array([5000], dtype=np.float32),
            np.arange(12).reshape(2, 2, 3),
            np.array([0.1, 0, -0.5]),
            np.array([0.2, 0, 0.5]),
        )
        self.received_action = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.raw, {"reset": True}

    def step(self, action):
        self.received_action = action
        return self.raw, 1.5, False, True, {"step": True}


@pytest.mark.parametrize("images_only", [False, True])
def test_observation_matches_declared_space_and_render(images_only):
    env = SyntheticTMRLEnv()
    wrapped = TMRLFullObsWrapper(env, use_only_images=images_only)
    assert wrapped.render() is None
    observation, info = wrapped.reset()
    assert info == {"reset": True}
    assert wrapped.observation_space.contains(observation)
    images = observation if images_only else observation["image_history"]
    np.testing.assert_array_equal(images, env.raw[3])
    assert images.dtype == np.uint8
    if not images_only:
        np.testing.assert_allclose(
            observation["action_history"], [0.1, 0, -0.5, 0.2, 0, 0.5]
        )
        assert observation["action_history"].dtype == np.float32
    frame = wrapped.render()
    expected = (
        np.repeat(images[-1, :, :, None], 3, axis=2).repeat(4, axis=0).repeat(4, axis=1)
    )
    np.testing.assert_array_equal(frame, expected)
    assert frame.shape == (8, 12, 3)
    assert wrapped.render_mode == "rgb_array"
    assert wrapped.metadata["render_fps"] == 10
    assert wrapped.metadata["source"] == "synthetic"
    assert "render_modes" not in env.metadata


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ([0.75, -0.2], [0.75, 0, -0.2]),
        ([-0.5, 0.3], [0, 0.5, 0.3]),
        ([0, 1], [0, 0, 1]),
    ],
)
def test_longitudinal_action_is_mapped_during_step(action, expected):
    env = SyntheticTMRLEnv()
    wrapped = TMRLFullObsWrapper(env, longitudinal_axis=True)
    observation, reward, terminated, truncated, info = wrapped.step(np.array(action))
    np.testing.assert_allclose(env.received_action, expected)
    assert env.received_action.dtype == np.float32
    assert wrapped.action_space.shape == (2,)
    assert wrapped.observation_space.contains(observation)
    assert (reward, terminated, truncated, info) == (1.5, False, True, {"step": True})


def test_default_action_is_passed_through():
    env = SyntheticTMRLEnv()
    wrapped = TMRLFullObsWrapper(env)
    action = np.array([0.2, 0.3, -0.4], dtype=np.float32)
    wrapped.step(action)
    np.testing.assert_array_equal(env.received_action, action)
    assert wrapped.action_space == env.action_space


def test_non_tuple_observation_space_is_rejected():
    env = SyntheticTMRLEnv()
    env.observation_space = spaces.Box(0, 1, shape=(3,))
    with pytest.raises(TypeError, match="full observation space"):
        TMRLFullObsWrapper(env)
