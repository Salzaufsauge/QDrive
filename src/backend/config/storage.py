import copy
from pathlib import Path

import yaml

from backend.config.config import ExperimentConfig
from util.utils import get_config_path, get_project_root, make_model_path


def load_config(config_path: Path):
    path = get_project_root() / config_path

    if not path.exists():
        raise FileNotFoundError(f"Config file {path} not found")

    return ExperimentConfig(yaml.safe_load(path.read_text()), config_path=config_path)


def save_model(config: ExperimentConfig, model):
    model_path = config.abs_model_path
    vecnorm = model.get_vec_normalize_env()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if vecnorm is not None:
        vecnorm_path = str(model_path).replace(".zip", ".pkl")
        vecnorm.save(vecnorm_path)
    model.save(model_path)


def save_config(config: ExperimentConfig):
    cfg_path = config.abs_config_path
    cfg = yaml.safe_dump(config.config, sort_keys=False)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w") as f:
        f.write(cfg)


def checkpoint_path(config: ExperimentConfig) -> Path:
    base_path = config.abs_model_path
    return base_path.with_name(base_path.stem + "_last.zip")


def replay_buffer_path(config: ExperimentConfig) -> Path:
    base_path = config.abs_model_path
    return base_path.with_name(base_path.stem + "_replay_buffer.pkl")


def save_checkpoint(config: ExperimentConfig, model, train_start_timesteps: int = 0):
    path = checkpoint_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.save(path)

    vecnorm = model.get_vec_normalize_env()
    if vecnorm is not None:
        vecnorm.save(str(path).replace(".zip", ".pkl"))

    config.config["last_timesteps"] = train_start_timesteps + int(model.num_timesteps)
    save_config(config)

    if not config.config.get("save_replay_buffer", True):
        return
    if not hasattr(model, "save_replay_buffer"):
        return

    buffer_path = replay_buffer_path(config)
    tmp_path = buffer_path.with_name(buffer_path.stem + ".tmp")
    model.save_replay_buffer(tmp_path)
    tmp_path.replace(buffer_path)


def as_new_run(config: ExperimentConfig) -> ExperimentConfig:
    cfg = copy.deepcopy(config.config)
    cfg.pop("current_timesteps", None)
    cfg.pop("last_timesteps", None)
    cfg.pop("best_reward", None)
    cfg["model_path"] = make_model_path(
        cfg["env_param"]["env_id"], cfg["algorithm"], cfg["model_param"]["policy"]
    )
    return ExperimentConfig(cfg, get_config_path(cfg["model_path"]))
