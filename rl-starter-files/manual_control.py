import gymnasium as gym
from minigrid_saywhat import *
from minigrid.manual_control import ManualControl

if __name__ == "__main__":
    env: MiniGridEnv = gym.make("SayWhat-FindEveryColorObj-v0")

    manual_control = ManualControl(env, agent_view=True)
    manual_control.start()