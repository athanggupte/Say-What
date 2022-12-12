import gymnasium as gym


def make_env(env_key, seed=None, render_mode=None, open_all_doors=False):
    if "-GoTo-" in env_key:
        env = gym.make(env_key, render_mode=render_mode, doors_open=open_all_doors)
    else:
        env = gym.make(env_key, render_mode=render_mode)
    env.reset(seed=seed)
    return env
