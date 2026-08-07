from pathlib import Path

import yaml

from backend.config.config import ExperimentConfig
from util.utils import get_project_root


def load_config(config_path: Path):
    path = get_project_root() / config_path

    if not path.exists():
        raise FileNotFoundError(f"Config file {path} not found")

    return ExperimentConfig(yaml.safe_load(path.read_text()), config_path=config_path)


def save_model(config: ExperimentConfig, model):
    model_path = config.abs_model_path
    vecnorm = model.get_vec_normalize_env()
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
