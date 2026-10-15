from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from util import utils


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("models/Env/Algo/model.zip", "experiments/Env/Algo/model.yaml"),
        ("other/models/Env/model.zip", "other/experiments/Env/model.yaml"),
        ("saved_models/model.zip", "saved_models/model.yaml"),
    ],
)
def test_config_path_replaces_directory_component(model, expected):
    assert utils.get_config_path(model) == Path(expected)


def test_empty_string_conversion_preserves_false_and_zero():
    raw = {"a": "", "nested": [{"b": ""}, 0, False, None, "text"]}
    assert utils.replace_empty_strings(raw) == {
        "a": None,
        "nested": [{"b": None}, 0, False, None, "text"],
    }
    assert raw["nested"][0]["b"] == ""


@pytest.mark.parametrize(
    "noise_type", ["NormalActionNoise", "OrnsteinUhlenbeckActionNoise"]
)
def test_action_noise_broadcasts_parameters_without_mutation(noise_type):
    spec = {"type": noise_type, "mean": "0.2", "sigma": "0.3"}
    env = SimpleNamespace(action_space=SimpleNamespace(shape=(3,)))
    action_noise = utils.build_action_noise(spec, env)
    np.testing.assert_allclose(action_noise._mu, [0.2] * 3)
    np.testing.assert_allclose(action_noise._sigma, [0.3] * 3)
    assert spec == {"type": noise_type, "mean": "0.2", "sigma": "0.3"}


@pytest.mark.parametrize("spec", [None, {}, {"sigma": 0.2}])
def test_unspecified_action_noise_is_disabled(spec):
    assert utils.build_action_noise(spec, None) is None


def test_unknown_action_noise_is_rejected():
    with pytest.raises(ValueError, match="Unknown noise type"):
        utils.build_action_noise({"type": "Typo"}, None)


def test_normalization_stats_are_independent_copies(monkeypatch):
    source = SimpleNamespace(
        obs_rms={"mean": np.array([1.0, 2.0])},
        ret_rms=SimpleNamespace(mean=np.array([3.0])),
    )
    target = SimpleNamespace(training=False, norm_reward=False)
    monkeypatch.setattr(utils, "unwrap_vec_normalize", lambda env: target)
    model = SimpleNamespace(get_vec_normalize_env=lambda: source)
    utils.copy_vecnorm(model, object())
    np.testing.assert_array_equal(target.obs_rms["mean"], [1, 2])
    np.testing.assert_array_equal(target.ret_rms.mean, [3])
    target.obs_rms["mean"][0] = 99
    target.ret_rms.mean[0] = 99
    assert source.obs_rms["mean"][0] == 1
    assert source.ret_rms.mean[0] == 3
    assert target.training is False
    assert target.norm_reward is False


@pytest.mark.parametrize("text", ["plain text", "\x1b[31mplain text\x1b[0m"])
def test_cleanup_ansi_preserves_text(text):
    assert utils.cleanup_ansi(text) == "plain text"
