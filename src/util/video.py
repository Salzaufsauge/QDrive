from stable_baselines3.common.vec_env import VecVideoRecorder

import wandb
from util.utils import get_project_root, log


def record_and_upload_video(
    trainer,
    model,
    eval_env,
    video_path,
    caption,
    step,
    video_length=2000,
    history_step=None,
):
    video_step = int(step)
    history_step = video_step if history_step is None else int(history_step)
    rec_env = VecVideoRecorder(
        eval_env,
        str(video_path),
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=str(video_step) + "-",
    )
    obs = rec_env.reset()
    for _ in range(video_length):
        action, _states = model.predict(obs, deterministic=True)
        obs, _rewards, _dones, _info = rec_env.step(action)
        rec_env.render()
    rec_env.close()

    video = wandb.Video(rec_env.video_path, caption=caption, format="mp4")
    trainer.run.log({"video_step": video_step}, step=history_step, commit=False)
    trainer.run.log({"video": video}, step=history_step, commit=True)


def record_pending_best_model(trainer, eval_env, history_step=None):
    pending = trainer.pending_best_model
    if pending is None:
        return

    log("INFO", f"Recording video for best model: (reward={pending['reward']:.2f})")
    model_class = trainer.algorithms.get(trainer.config.algorithm)
    best_model = model_class.load(pending["model_path"], env=eval_env)

    vecnorm_path = str(pending["model_path"]).replace(".zip", ".pkl")
    vecnorm = best_model.get_vec_normalize_env()
    if vecnorm is not None:
        loaded = vecnorm.load(vecnorm_path, venv=vecnorm)
        vecnorm.obs_rms = loaded.obs_rms
        vecnorm.ret_rms = loaded.ret_rms

    video_path = (
        get_project_root()
        / trainer.config.model_path.replace("models", "experiments").replace(".zip", "")
        / "best"
    )
    record_and_upload_video(
        trainer,
        best_model,
        eval_env,
        video_path,
        caption=f"best model so far, reward={pending['reward']:.2f}",
        step=pending["timesteps"],
        history_step=history_step,
    )
    trainer.pending_best_model = None
