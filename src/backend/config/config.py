from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from util.inspection_helper import parse_params
from util.utils import get_project_root


@dataclass(frozen=False)
class ExperimentConfig:
    config: dict
    config_path: Path

    @property
    def env_params(self):
        return self.config["env_param"]

    @property
    def env_wrappers(self):
        return self.config["env_wrappers"]

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

    @property
    def callback_params(self):
        return self.config["callback_params"]

    @property
    def best_reward(self):
        return self.config.get("best_reward", float("-inf"))

    @best_reward.setter
    def best_reward(self, value):
        self.config["best_reward"] = value
