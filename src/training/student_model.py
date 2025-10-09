import torch
import torch.nn as nn
from typing import List

class StudentModel(nn.Module):
    """
    A simple feed-forward neural network to act as the 'student' in
    a knowledge distillation setup.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], dropout: float = 0.2):
        super(StudentModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 