from .findall import *

from gymnasium.envs.registration import register

register(
    id="SayWhat-FindEveryObj-v0",
    entry_point="minigrid_saywhat.findall:FindEveryObj"
)

register(
    id="SayWhat-FindEveryObj_Oracle-v0",
    entry_point="minigrid_saywhat.findall:FindEveryObj",
    kwargs={"use_oracle": True}
)