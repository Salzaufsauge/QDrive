import gymnasium as gym

def get_envs():
    return sorted([env_id for env_id in gym.envs.registry.keys()])