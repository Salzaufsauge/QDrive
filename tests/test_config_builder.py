import inspect
from pathlib import Path

import pytest

from backend.config import builder


def model_signature(
    policy,
    count: int,
    optional: int | None = None,
    rate: float | int = 0.1,  # noqa: PYI041 - exercise mixed numeric annotations
    options: dict | None = None,
):
    pass


def test_signature_filters_and_converts_editor_values():
    raw = {
        "policy": "MlpPolicy",
        "count": "12",
        "optional": "3",
        "rate": "0.25",
        "options": {
            "width": "[64, 32]",
            "enabled": "True",
            "empty": "",
            "missing": None,
            "": "ignored",
        },
        "unknown": "discard",
    }
    result = builder.build_config(raw, inspect.signature(model_signature).parameters)
    assert result == {
        "policy": "MlpPolicy",
        "count": 12,
        "optional": 3,
        "rate": 0.25,
        "options": {"width": [64, 32], "enabled": True},
    }
    assert raw["options"]["width"] == "[64, 32]"


def test_missing_values_are_omitted():
    assert (
        builder.build_config(
            {"count": None}, inspect.signature(model_signature).parameters
        )
        == {}
    )


def test_invalid_integer_is_rejected():
    with pytest.raises(ValueError):
        builder.build_config(
            {"count": "many"}, inspect.signature(model_signature).parameters
        )


@pytest.fixture
def raw_config(monkeypatch):
    def algorithm(policy, batch_size: int = 64):
        pass

    def wrapper(env, size: int = 4):
        pass

    monkeypatch.setattr(builder, "load_algorithms", lambda: {"TestAlgo": algorithm})
    monkeypatch.setattr(builder, "load_env_wrappers", lambda: {"TestWrapper": wrapper})
    monkeypatch.setattr(
        builder, "make_model_path", lambda *args: "models/Test-v0/TestAlgo/model.zip"
    )
    return {
        "algorithm": "TestAlgo",
        "env_param": {"env_id": "Test-v0", "n_envs": "2"},
        "env_wrappers": {"TestWrapper": {"size": "8"}},
        "model_param": {"policy": "MlpPolicy", "batch_size": "32"},
        "milestones": ["100", "20", "3"],
        "notes": "",
    }


def test_build_converts_sections_and_derives_paths(raw_config):
    config = builder.ConfigBuilder.build(raw_config)
    assert config.env_params == {"env_id": "Test-v0", "n_envs": 2}
    assert config.env_wrappers == {"TestWrapper": {"size": 8}}
    assert config.config["model_param"] == {"policy": "MlpPolicy", "batch_size": 32}
    assert config.milestones == ["3", "20", "100"]
    assert config.model_path == "models/Test-v0/TestAlgo/model.zip"
    assert config.config_path == Path("experiments/Test-v0/TestAlgo/model.yaml")
    assert config.config["notes"] is None
    assert raw_config["env_param"]["n_envs"] == "2"


def test_empty_milestones(raw_config):
    raw_config["milestones"] = []
    assert builder.ConfigBuilder.build(raw_config).milestones is None


def test_build_reports_cause_for_unknown_wrapper(raw_config):
    raw_config["env_wrappers"] = {"Missing": {}}
    with pytest.raises(RuntimeError, match="Error while building config") as error:
        builder.ConfigBuilder.build(raw_config)
    assert isinstance(error.value.__cause__, KeyError)
