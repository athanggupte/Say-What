import torch
import torch.nn as nn

from .model import Critic

class Critic1(Critic):
    def __init__(self, input_dim, output_dim, device):
        super().__init__(input_dim, output_dim, device)
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 128
        self.layers = 4

        self.input_layer = nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim, device=device)
        self.mlp = nn.ModuleList([
                        nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, device=device),
                        nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, device=device),
                        nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, device=device),
                    ])
        self.output_layer = nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim, device=device)

    def forward(self, x):
        y = self.input_layer(x)
        for l in self.mlp:
            y = l(y)
        y = self.output_layer(y)
        return y
