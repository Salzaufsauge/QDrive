import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def main():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    env_args = dict(domain_randomize=True, continuous=True, render_mode=None)

    env = make_vec_env("CarRacing-v3", n_envs=8, env_kwargs=env_args)

    model: PPO = PPO("CnnPolicy", env, batch_size=1024, n_steps=4096, verbose=1)
    model.learn(total_timesteps=20_000, progress_bar=True)
    model.save(f"../models/{env.envs[0].spec.id}/model-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

    env.close()

    env_args["render_mode"] = "human"

    eval_env  = make_vec_env("CarRacing-v3", n_envs=1, env_kwargs=env_args)
    obs = eval_env.reset()

    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward,terminated, info = eval_env.step(action)

        eval_env.render()

    eval_env.close()







if __name__ == "__main__":
    main()