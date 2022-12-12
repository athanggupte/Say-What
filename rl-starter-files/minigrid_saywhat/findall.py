from minigrid.envs.babyai.core.verifier import *
from minigrid.envs.babyai.goto import *

class FindEveryInstr(ActionInstr):
    """
    Go next to (and look towards) each object matching a given description
    Every matching object that is successfully "found" is deleted from the grid
    eg: find every red door
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
        objs_to_remove = []
        for idx, pos in enumerate(self.desc.obj_poss):
            # If the agent is next to (and facing) the object
            if np.array_equal(pos, self.env.front_pos):
                if len(self.desc.obj_poss) == 1:
                    return "success"
                else:
                    self.env.grid.set(*pos, None)
                    objs_to_remove.append(idx)
        for idx in objs_to_remove:
            self.desc.obj_poss.pop(idx)
            self.desc.obj_set.pop(idx)
        if self.env.use_oracle:
            self.env.oracle.update_subgoal()

        return "continue"


class Oracle:
    def __init__(self, env):
        self.env = env
        self.subgoal_stack = []

    def update_subgoal(self):
        objs = self.env.instrs.desc.obj_set
        poss = self.env.instrs.desc.obj_poss
        agent_pos = self.env.agent_pos
        closest = -1
        min_dist = 1000
        for i, pos in enumerate(poss):
            dist = abs(agent_pos[0] - pos[0]) + abs(agent_pos[1] - pos[1])
            if dist < min_dist:
                min_dist = dist
                closest = i
        obj = objs[closest]
        subgoal = GoToInstr(ObjDesc(obj.type, obj.color))
        self.subgoal_stack.append(subgoal)

    def get_subgoal(self):
        return self.subgoal_stack[-1]


class FindEveryObj(RoomGridLevel):
    """
    Go to an object. Single room. Many distractors.
    """

    def __init__(
        self,
        room_size=8,
        num_rows=1,
        num_cols=1,
        num_dists=8,
        doors_open=True,
        use_oracle=False,
        **kwargs,
    ):
        self.num_dists = num_dists
        self.doors_open = doors_open
        super().__init__(
            num_rows=num_rows, num_cols=num_cols, room_size=room_size, **kwargs
        )
        self.use_oracle = use_oracle

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = FindEveryInstr(ObjDesc(obj.type))

        # If requested, open all the doors
        if self.doors_open:
            self.open_all_doors()

        if self.use_oracle:
            self.oracle = Oracle(self)

    def step(self, action):
        if self.use_oracle:
            if action == self.actions.done:
                # action : DONE
                if self.oracle.subgoal_stack:
                    self.oracle.subgoal_stack.pop(-1)
            if action == self.action_space.n:
                # action : SUB
                self.oracle.update_subgoal()
                action = self.actions.done

        obs, reward, terminated, truncated, info = super().step(action)
        if self.use_oracle and self.oracle.subgoal_stack:
            obs["mission"] = self.oracle.get_subgoal().surface(self) + ". " + obs["mission"]

        return obs, reward, terminated, truncated, info

