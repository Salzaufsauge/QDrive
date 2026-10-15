from types import SimpleNamespace

import gymnasium as gym
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper

from backend.env import env_manager
from backend.env.env_manager import EnvMode
from backend.env.tmrl_env import TMRL_ENV_ID


@pytest.mark.parametrize(
    ("mode", "env_id", "expected_count", "expected_class"),
    [
        (EnvMode.TRAIN, "Test-v0", 4, SubprocVecEnv),
        (EnvMode.EVAL, "Test-v0", 1, DummyVecEnv),
        (EnvMode.TRAIN, TMRL_ENV_ID, 1, DummyVecEnv),
        (EnvMode.EVAL, TMRL_ENV_ID, 1, DummyVecEnv),
    ],
)
def test_vectorization_rules_preserve_config(
    monkeypatch, mode, env_id, expected_count, expected_class
):
    params = {"env_id": env_id, "n_envs": 4, "vec_env_cls": "SubprocVecEnv", "seed": 12}
    config = SimpleNamespace(env_params=params, env_wrappers={})
    captured = {}
    sentinel_env = object()

    def make_env(**kwargs):
        captured.update(kwargs)
        return sentinel_env

    monkeypatch.setattr(env_manager, "build_wrapper", lambda *args: ([], []))
    monkeypatch.setattr(env_manager, "make_vec_env", make_env)
    assert env_manager.build_env(config, mode) is sentinel_env
    assert captured == {
        "env_id": env_id,
        "n_envs": expected_count,
        "vec_env_cls": expected_class,
        "seed": 12,
        "wrapper_class": None,
    }
    assert params["n_envs"] == 4
    assert params["vec_env_cls"] == "SubprocVecEnv"


def test_wrapper_partition_overrides_and_order(monkeypatch):
    class GymWrapper(gym.Wrapper):
        pass

    class VectorWrapper(VecEnvWrapper):
        pass

    monkeypatch.setattr(
        env_manager,
        "load_env_wrappers",
        lambda: {"Gym": GymWrapper, "Vector": VectorWrapper},
    )
    monkeypatch.setattr(
        env_manager,
        "load_overrides",
        lambda: {"eval": {"env_wrappers": {"Vector": {"training": False}}}},
    )
    raw = {"Vector": {"training": True, "norm_obs": True}, "Gym": {"size": 4}}
    gym_wrappers, vec_wrappers = env_manager.build_wrapper(raw, EnvMode.EVAL)
    assert gym_wrappers[0].func is GymWrapper
    assert gym_wrappers[0].keywords == {"size": 4}
    assert vec_wrappers[0].func is VectorWrapper
    assert vec_wrappers[0].keywords == {"training": False, "norm_obs": True}
    assert raw["Vector"]["training"] is True


def test_gym_wrappers_compose_in_configuration_order():
    wrapped = env_manager.compose_gym_wrappers(
        [
            lambda env: ("first", env),
            lambda env: ("second", env),
        ]
    )
    assert wrapped("base") == ("second", ("first", "base"))


def test_invalid_wrapper_type_is_rejected(monkeypatch):
    monkeypatch.setattr(env_manager, "load_env_wrappers", lambda: {"Invalid": object})
    monkeypatch.setattr(
        env_manager, "load_overrides", lambda: {"train": {"env_wrappers": {}}}
    )
    with pytest.raises(TypeError, match="Unknown wrapper type"):
        env_manager.build_wrapper({"Invalid": {}}, EnvMode.TRAIN)
