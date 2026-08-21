import math
from numbers import Real
from typing import Any

from stable_baselines3.common.logger import KVWriter

_GLOBAL_STEP_PATTERNS = (
    "rollout/*",
    "time/*",
    "train/*",
    "eval/*",
    "episode/*",
    "milestone/*",
)


def configure_wandb_metrics(run: Any) -> None:
    run.define_metric("global_step")
    for pattern in _GLOBAL_STEP_PATTERNS:
        run.define_metric(pattern, step_metric="global_step", step_sync=False)

    run.define_metric("video_step")
    run.define_metric("video", step_metric="video_step")


def _as_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return None

    if hasattr(value, "item"):
        value = value.item()

    if not isinstance(value, Real):
        return None

    value = float(value)
    return value if math.isfinite(value) else None


def log_wandb_metrics(run: Any, global_step: int, metrics: dict[str, Any]) -> None:
    values = {
        key: scalar
        for key, value in metrics.items()
        if (scalar := _as_scalar(value)) is not None
    }
    if values:
        run.log({**values, "global_step": int(global_step)})


class WandbOutputFormat(KVWriter):
    def __init__(self, run: Any, step_offset: int = 0):
        self.run = run
        self.step_offset = step_offset

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        metrics = {
            key: value
            for key, value in key_values.items()
            if "wandb" not in key_excluded.get(key, ())
        }
        log_wandb_metrics(self.run, self.step_offset + step, metrics)

    def close(self) -> None:
        pass
