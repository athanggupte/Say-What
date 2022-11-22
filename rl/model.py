import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass