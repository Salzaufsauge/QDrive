from pathlib import Path

import pytest
import yaml

from backend.config import config as config_module
from backend.config import storage
from backend.config.config import ExperimentConfig


class FileModel:
    num_timesteps = 25

    def __init__(self, normalized=False):
        self.normalized = normalized

    def save(self, path):
        Path(path).write_bytes(b"model")

    def get_vec_normalize_env(self):
        return self if self.normalized else None

    def save_replay_buffer(self, path):
        Path(path).write_bytes(b"replay")


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(storage, "get_project_root", lambda: tmp_path)
    return ExperimentConfig(
        {
            "model_path": "models/Test-v0/Algo/model.zip",
            "algorithm": "Algo",
            "env_param": {"env_id": "Test-v0"},
            "model_param": {"policy": "MlpPolicy"},
            "milestones": [100],
            "current_timesteps": 10,
            "last_timesteps": 20,
            "best_reward": 4.5,
        },
        Path("experiments/Test-v0/Algo/model.yaml"),
    )


def test_yaml_round_trip(config):
    storage.save_config(config)
    loaded = storage.load_config(config.config_path)
    assert loaded.config == config.config
    assert loaded.config_path == config.config_path


def test_missing_config_has_useful_error(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "get_project_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        storage.load_config(Path("missing.yaml"))


@pytest.mark.parametrize("normalized", [False, True])
def test_best_model_and_optional_normalization_artifacts(config, normalized):
    storage.save_model(config, FileModel(normalized))
    assert config.abs_model_path.read_bytes() == b"model"
    assert config.abs_model_path.with_suffix(".pkl").exists() == normalized


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("save_replay", [False, True])
def test_checkpoint_saves_progress_and_optional_artifacts(
    config, normalized, save_replay
):
    config.config["save_replay_buffer"] = save_replay
    storage.save_checkpoint(config, FileModel(normalized), train_start_timesteps=100)
    checkpoint = config.abs_model_path.with_name("model_last.zip")
    replay = config.abs_model_path.with_name("model_replay_buffer.pkl")
    assert checkpoint.read_bytes() == b"model"
    assert checkpoint.with_suffix(".pkl").exists() == normalized
    assert replay.exists() == save_replay
    if save_replay:
        assert replay.read_bytes() == b"replay"
    assert not replay.with_suffix(".tmp").exists()
    assert config.config["last_timesteps"] == 125
    assert yaml.safe_load(config.abs_config_path.read_text())["last_timesteps"] == 125
    assert not config.abs_model_path.exists()


def test_checkpoint_supports_models_without_replay_buffers(config):
    class OnPolicyModel:
        num_timesteps = 5
        save = FileModel.save

        def get_vec_normalize_env(self):
            return None

    storage.save_checkpoint(config, OnPolicyModel())
    assert storage.checkpoint_path(config).exists()
    assert not storage.replay_buffer_path(config).exists()


def test_failed_replay_save_preserves_previous_buffer(config):
    class FailingModel(FileModel):
        def save_replay_buffer(self, path):
            Path(path).write_bytes(b"partial")
            raise OSError("disk full")

    replay = storage.replay_buffer_path(config)
    replay.parent.mkdir(parents=True)
    replay.write_bytes(b"previous")
    with pytest.raises(OSError, match="disk full"):
        storage.save_checkpoint(config, FailingModel())
    assert replay.read_bytes() == b"previous"


def test_new_run_resets_progress_and_deep_copies_config(config, monkeypatch):
    monkeypatch.setattr(
        storage, "make_model_path", lambda *args: "models/new/model.zip"
    )
    new = storage.as_new_run(config)
    assert (
        not {"current_timesteps", "last_timesteps", "best_reward"} & new.config.keys()
    )
    assert new.model_path == "models/new/model.zip"
    assert new.config_path == Path("experiments/new/model.yaml")
    new.config["env_param"]["env_id"] = "Changed-v0"
    new.milestones.append(200)
    assert config.env_params["env_id"] == "Test-v0"
    assert config.milestones == [100]
    assert config.config["last_timesteps"] == 20
