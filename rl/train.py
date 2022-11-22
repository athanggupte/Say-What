import gymnasium as gym
from rl import *

env = gym.make('CartPole-v1')

agent = DiscreteSACAgent(env, Actor1, Critic1, Critic1,
                         actor_lr=0.0005, critic_lr=0.0005, batch_size=100, discount=0.99,
                         critic_tau=0.01, temperature_lr=0.0005, init_temperature=1.0, learn_temperature=True,
                         replay_buffer_capacity=1000,
                         device=torch.device('cuda'))


NUM_EPISODES = 400
STEPS_PER_EPISODE = 200

for ep in range(NUM_EPISODES):
    state, _ = env.reset()

    done = False
    i = 0
    episode_reward = 0

    while not done and i < STEPS_PER_EPISODE:
        i += 1
        action = agent.get_next_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.update_transition(state, action, reward, next_state, done, i)

        episode_reward += reward

        state = next_state

    print(episode_reward)
