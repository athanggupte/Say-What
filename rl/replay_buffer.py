import numpy as np
import torch

class DiscreteReplayBuffer():
    """
    Buffer to store episodes for replay sampling.
    Modified version of ReplayBuffer from https://github.dev/denisyarats/pytorch_sac/blob/master/agent/replay_buffer.py

    replay buffer (D) = [(s, a, r, s', d)]
    """

    def __init__(self, obs_shape, obs_dtype, capacity, device):
        self.device = device
        self.capacity = capacity

        # Initialize buffer to empty
        self.obs = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, 1), dtype=np.int32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_obs = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add_transition(self, obs, action, reward, next_obs, done):
        np.copyto(self.obs[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obs[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        assert(self.__len__() >= batch_size)
        idxs = np.random.randint(0, self.__len__(), size=batch_size)

        obs = torch.as_tensor(self.obs[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device).long()
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device).float()
        next_obs = torch.as_tensor(self.next_obs[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device).float()

        return obs, actions, rewards, next_obs, not_dones