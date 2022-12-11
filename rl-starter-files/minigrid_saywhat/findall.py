from minigrid.envs.babyai.core.verifier import *
from minigrid.envs.babyai.goto import *

class FindEveryInstr(ActionInstr):
    """
    Go next to (and look towards) an object matching a given description
    eg: go to the door
    """

    def __init__(self, obj_desc):
        super().__init__()
        self.desc = obj_desc

    def surface(self, env):
        import re
        obj_desc = self.desc.surface(env)
        obj_desc = re.sub(r"\ba\b", "", obj_desc)
        obj_desc = re.sub(r"\bthe\b", "", obj_desc)
        return "find every" + obj_desc

    def reset_verifier(self, env):
        super().reset_verifier(env)

        # Identify set of possible matching objects in the environment
        self.desc.find_matching_objs(env)

    def verify_action(self, action):
        # For each object position
        if len(self.desc.obj_poss) == 1:
            for pos in self.desc.obj_poss:
                # If the agent is next to (and facing) the object
                if np.array_equal(pos, self.env.front_pos):
                    return "success"

        return "continue"


class FindEveryColorObj(RoomGridLevel):
    """
    Go to an object, the object may be in another room. Many distractors.
    """

    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        doors_open=True,
        **kwargs,
    ):
        self.num_dists = num_dists
        self.doors_open = doors_open
        super().__init__(
            num_rows=num_rows, num_cols=num_cols, room_size=room_size, **kwargs
        )

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = FindEveryInstr(ObjDesc(obj.type, obj.color))

        # If requested, open all the doors
        if self.doors_open:
            self.open_all_doors()