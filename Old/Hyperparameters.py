from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class Hyperparameters:
    x_dimension: int

    nn_width: int
    nn_activation: str

    dal_num_iterations: int

    max_error: float

class BasicNN(nn.Module):
    def __init__(self, input_dim, width, activation):
        super(BasicNN, self).__init__()

        self.layer = nn.Linear(in_features=input_dim, out_features=width)
        self.activation = nn.ReLU
        self.final_layer = nn.Linear(in_features=width, out_features=1)

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        x = self.final_layer(x)
        return x



# This distribution is for both x and y
class Distribution(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, n):
        # returns n of (x, y) pairs
        pass

    @abstractmethod
    def get_probability(self, x: np.array) -> float:
        pass


class SamplingProcedure(ABC):
    def __init__(self):
        pass

    def