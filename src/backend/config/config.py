from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from util.utils import get_project_root, parse_params


@dataclass(frozen=False)
class ExperimentConfig:
    config: dict
    config_path: Path

    @property
    def env_params(self):
        return self.config["env_param"]

    @property
    def vec_frame_stack(self):
        return self.config["vec_frame_stack"]

    @property
    def algorithm(self):
        return self.config["algorithm"]

    @property
    def model_path(self):
        return self.config["model_path"]

    @property
    def abs_model_path(self):
        return get_project_root() / self.model_path

    @property
    def abs_config_path(self):
        return get_project_root() / self.config_path

    @cached_property
    def model_params(self):
        return parse_params(self.config["model_param"])

    @property
    def milestones(self):
        return self.config["milestones"]

    @milestones.setter
    def milestones(self, value):
        self.config["milestones"] = value

    @property
    def total_timesteps(self):
        return self.config["total_timesteps"]

    @property
    def current_timesteps(self):
        return self.config["current_timesteps"]

    @current_timesteps.setter
    def current_timesteps(self, value):
        self.config["current_timesteps"] = value