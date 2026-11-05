"""Feed-forward neural network for power plant energy regression."""

import torch
from torch import nn


class PowerPlantANN(nn.Module):
    """A compact ANN with architecture 4 -> 6 -> 6 -> 1."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 6), nn.ReLU(),
            nn.Linear(6, 6), nn.ReLU(),
            nn.Linear(6, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
