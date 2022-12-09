from bot import *
from minigrid.wrappers import *
import matplotlib.pyplot as plt
import copy 
import random

env = gym.make('BabyAI-BossLevel-v0', render_mode='rgb_array')
env = RGBImgPartialObsWrapper(env)
env.reset()
img = env.render()
plt.imshow(img)
expert = Bot(env)


def stacking(expert):
    print(expert.stack)

def create_subgoal_stack(expert, env):
    subgoal_stack=[]
    print(env.mission)
    expert = Bot(env)
    print(expert.stack)
    i=0
    while i < len(expert.stack):
        subgoal, skip = _process_subgoal(expert,expert.stack[i],i)
        subgoal_stack.append(subgoal)
        i=i+skip
    return subgoal_stack

# def get_data(env):
#     action = None
#     actions =[]
#     subgoal_stacks = []
#     while action != env.actions.done:
#         action = expert.replan()
#         #print(action)
#         subgoal_stack = create_subgoal_stack(expert, env)
#         subgoal_stacks.append(subgoal_stack)
#         actions.append(action)
#         env.step(action)
#         #print(env)
#     print('DONE')
#     return actions, subgoal_stacks, env.mission

def get_data(env):
    action=None
    actions =[]
    subgoal_stacks = []
    done = ""
    count = 0
    while not (done or action == env.actions.done):
        if(count%15 == 0):
            print("------------------------------START BOT-------------------------------")
            new_env_subgoal_stacks=[]
            new_env_action = None
            new_env_actions=[]
            new_env_done=""
            new_env = copy.deepcopy(env) 
            expert=Bot(new_env)
            print(env)
            while not (new_env_done or new_env_action == new_env.actions.done):
                new_env_action = expert.replan()
                subgoal_stack = create_subgoal_stack(expert, env)
                new_env_subgoal_stacks.append(subgoal_stack)
                new_env_actions.append(new_env_action)
                obs, rew, new_env_done, _, info = new_env.step(new_env_action)
            print("BOT DONE : ", str(new_env_done))
            print("------------------------------END BOT-------------------------------")

        actions.append(new_env_actions)
        subgoal_stacks.append(new_env_subgoal_stacks)
        action = random.choice([env.actions.forward, env.actions.left, env.actions.toggle, env.actions.right])
        print(action)
        observation, reward, done, _ ,info = env.step(action)
        count=count+1
        if count > 30:
            break
    print('DONE')
    return actions, subgoal_stacks, env.mission

def look_ahead(expert,i):
    try:
        subgoal = expert.stack[i]
    except:
        return None
    return subgoal.datum

def _process_subgoal(expert, subgoal, index):
    # """
    # Translate subgoals into instruction form which the agent can execute
    # """
    

    if isinstance(subgoal, GoNextToSubgoal):
        return GoToInstr(subgoal.datum), 1
        

    if isinstance(subgoal, OpenSubgoal):
        ans = look_ahead(expert,index+1)
        return OpenInstr(ans), 2
        

    if isinstance(subgoal, PickupSubgoal):
        ans = look_ahead(expert,index+1)
        return PickupInstr(ans), 2

    
    if isinstance(subgoal, DropSubgoal):
        if(look_ahead(expert,index+3)):
            ans = look_ahead(expert,index+3)
            obj_fixed = expert.stack[index+1].datum
            return PutNextInstr(ans, obj_fixed), 4
        else:
            ans=look_ahead(expert,index+2)
            return PickupInstr(ans), 3



actions,ss,_ = get_data(env)

for s in ss:
    print(s)
# print(ss)

