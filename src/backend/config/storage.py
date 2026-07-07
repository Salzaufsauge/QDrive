from pathlib import Path

import yaml

from backend.config.config import ExperimentConfig
from util.utils import get_project_root


def load_config(config_path: Path):
    return yaml.safe_load(get_project_root() / config_path)


def save_model(config: ExperimentConfig, model):
    model_path = get_project_root() / config.model_path
    model.save(model_path)


def save_config(config: ExperimentConfig):
    cfg_path = config.abs_config_path
    cfg = yaml.safe_dump(config, sort_keys=False)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w") as f:
        f.write(cfg)
