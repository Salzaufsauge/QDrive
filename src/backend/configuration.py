from pathlib import Path

import yaml

from util.utils import get_project_root


def get_config_path(model_path):
    model_path = Path(model_path)
    parts = list(model_path.parts)

    if "models" in parts:
        i = parts.index("models")
        parts[i] = "experiments"

    conf_path = Path(*parts)
    conf_path = conf_path.with_suffix(".yaml")

    return conf_path


class Configuration:
    def __init__(self):
        self.config = dict()

    def load_model(self, model_path: Path):
        self.config = yaml.safe_load(get_config_path(get_project_root() / model_path).read_text())

    def save_model(self, model):
        model_path = get_project_root() / self.config["model_path"]
        model.save(model_path)
        cfg = yaml.safe_dump(self.config, sort_keys=False)
        cfg_path = get_config_path(model_path)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with cfg_path.open("w") as f:
            f.write(cfg)
