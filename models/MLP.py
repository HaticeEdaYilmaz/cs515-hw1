import torch.nn as nn  
import torch
from typing import List, Type

class MLP(nn.Module):
    """Simple multi-layer perceptron with optional dropout and batchnorm."""
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, dropout: float, activation: Type[nn.Module] = nn.ReLU, batchnorm: bool =False) -> None:
        super().__init__()
        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   # flatten (B, 1, 28, 28) → (B, 784)
        return self.net(x)

