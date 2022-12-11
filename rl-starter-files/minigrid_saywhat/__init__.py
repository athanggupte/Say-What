from .findall import *

from gymnasium.envs.registration import register

register(
    id="SayWhat-FindEveryColorObj-v0",
    entry_point="minigrid_saywhat.findall:FindEveryColorObj",
)