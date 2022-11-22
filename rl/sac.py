from gymnasium import Env
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type

from replay_buffer import DiscreteReplayBuffer
from utils import soft_update
from .model import Actor, Critic


class DiscreteSACAgent(nn.Module):
    """
    Soft Actor-Critic algorithm for Discrete spaces
    Adapted from https://github.dev/denisyarats/pytorch_sac/blob/master/agent/sac.py,
    https://towardsdatascience.com/adapting-soft-actor-critic-for-discrete-action-spaces-a20614d4a50a, and
    https://arxiv.org/pdf/1910.07207.pdf
    """

    def __init__(self, env: Env,
                 actor_class: Type[Actor], critic_1_class: Type[Critic], critic_2_class: Type[Critic],
                 actor_lr, critic_lr,
                 batch_size, discount, critic_tau,
                 temperature_lr, init_temperature, learn_temperature,
                 replay_buffer_capacity,
                 device):
        super().__init__()

        self.device = device

        # Save the environment config
        self.env = env
        self.action_dim = env.action_space.n
        self.state_dim = env.observation_space.shape[0]

        # Save the learning parameters
        self.batch_size = batch_size
        self.discount = discount
        self.critic_tau = critic_tau

        # Initialize the actor network
        self.actor = actor_class(input_dim=self.state_dim, output_dim=self.action_dim, device=device)

        # Initialize the dual critic networks
        self.critic_1 = critic_1_class(input_dim=self.state_dim, output_dim=self.action_dim, device=device)
        self.critic_2 = critic_2_class(input_dim=self.state_dim, output_dim=self.action_dim, device=device)
        
        # Initialize the target critics
        self.critic_1_target = critic_1_class(input_dim=self.state_dim, output_dim=self.action_dim, device=device)
        self.critic_2_target = critic_2_class(input_dim=self.state_dim, output_dim=self.action_dim, device=device)

        # Copy parameters from critic parameters to targets
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # Initialize alpha (temperature) : factor of entropy regularization on the gradient updates
        self.learn_temperature = learn_temperature
        self.log_alpha = torch.tensor(np.log(init_temperature), requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        self.target_entropy = -0.98 * np.log(1 / self.action_dim)

        # Initialize optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optim = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optim = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=temperature_lr)

        # Initialize replay buffer
        self.replay_buffer = DiscreteReplayBuffer(self.env.observation_space.shape,
                                                  self.env.observation_space.dtype,
                                                  replay_buffer_capacity,
                                                  device)

        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic_1.train(training)
        self.critic_2.train(training)

    def update_transition(self, obs, action, reward, next_obs, done, step):
        self.replay_buffer.add_transition(obs, action, reward, next_obs, done)
        if len(self.replay_buffer) >= self.batch_size:
            self.update(step)

    def update(self, step):
        obs, actions, rewards, next_obs, not_dones = self.replay_buffer.sample(self.batch_size)

        self.critic_1_optim.zero_grad()
        self.critic_2_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.alpha_optim.zero_grad()
        
        self.update_critics(obs, actions, rewards, next_obs, not_dones)
        self.update_actor_and_alpha(obs)
        self.update_targets()

    def update_critics(self, obs, actions, rewards, next_obs, not_dones):
        with torch.no_grad():
            # pi_theta(s'), log pi_theta(s') : (|B|, |A|)
            action_probs, log_action_probs = self.get_action_info(next_obs)

            # Q_phi_target1(s'), Q_phi_target2(s') : (|B|, |A|)
            target_Q1 = self.critic_1_target(next_obs)
            target_Q2 = self.critic_2_target(next_obs)

            # V(s') : (|B|, 1)
            target_V = (action_probs * (torch.min(target_Q1, target_Q2) - self.alpha * log_action_probs)).sum(dim=1)

            # y(r, s', d) : (|B|, )
            y = (rewards + not_dones * self.discount * target_V.unsqueeze(-1)).squeeze(-1)
        
        # Q_phi_1(s), Q_phi_2(s) : (|B|, )
        soft_Q1_values = self.critic_1(obs).gather(1, actions).squeeze(-1)
        soft_Q2_values = self.critic_2(obs).gather(1, actions).squeeze(-1)

        critic_1_error = torch.nn.MSELoss(reduction='none')(soft_Q1_values, y)
        critic_2_error = torch.nn.MSELoss(reduction='none')(soft_Q2_values, y)

        critic_1_loss = critic_1_error.mean()
        critic_2_loss = critic_2_error.mean()

        critic_1_loss.backward()
        critic_2_loss.backward()

        self.critic_1_optim.step()
        self.critic_2_optim.step()

    def update_actor_and_alpha(self, obs):
        # pi_theta(s), log pi_theta(s) : (|B|, |A|)
        action_probs, log_action_probs = self.get_action_info(obs)

        # Q_phi_1(s), Q_phi_2(s) : (|B|, |A|)
        soft_Q1 = self.critic_1(obs).detach()
        soft_Q2 = self.critic_2(obs).detach()

        # V(s') : (|B|, 1)
        soft_V = (action_probs * (self.alpha.detach() * log_action_probs - torch.min(soft_Q1, soft_Q2))).sum(dim=1)

        actor_loss = soft_V.mean()

        actor_loss.backward()
        self.actor_optim.step()

        # Temperature
        if self.learn_temperature:
            temperature_loss = -(self.log_alpha * (log_action_probs + self.target_entropy).detach()).mean()
            # temperature_loss = (action_probs.detach() * (self.target_entropy - self.alpha)).sum(dim=1).mean()
            temperature_loss.backward()
            self.alpha_optim.step()

    def update_targets(self):
        soft_update(self.critic_1, self.critic_1_target, self.critic_tau)
        soft_update(self.critic_2, self.critic_2_target, self.critic_tau)


    def get_action_info(self, obs):
        # print("obs : ", obs.shape)
        action_probs = self.actor(obs)
        log_action_probs = torch.log(action_probs + 1e-8)
        return action_probs, log_action_probs


    def get_next_action(self, obs, eval=False):
        if eval:
            action = self.get_deterministic_action(obs)
        else:
            action = self.get_probabilistic_action(obs)
        return action

    def get_deterministic_action(self, obs):
        action_probs = self.get_action_probabilities(obs)
        action = np.argmax(action_probs)
        return action

    def get_probabilistic_action(self, obs):
        action_probs = self.get_action_probabilities(obs)
        action = np.random.choice(np.arange(self.action_dim), p=action_probs)
        return action

    def get_action_probabilities(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float, device=self.device).unsqueeze(0)
        action_probs = self.actor(obs_tensor).squeeze(0)
        return action_probs.cpu().detach().numpy()

