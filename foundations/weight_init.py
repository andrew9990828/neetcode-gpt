import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        init = math.sqrt(2/(fan_in+fan_out))
        W = torch.randn(fan_out, fan_in) * init
        return torch.round(W, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        init = math.sqrt(2/(fan_in))
        W = torch.randn(fan_out, fan_in) * init
        return torch.round(W, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        # torch.manual_seed(0)
        # std_act = []
        # x = torch.randn(input_dim)

        torch.manual_seed(0)

        dims = [input_dim] + [hidden_dim] * num_layers
        weights = []

        # Generate ALL weights first
        for i in range(num_layers):
            fan_in = dims[i]
            fan_out = dims[i + 1]

            if init_type == "xavier":
                scale = math.sqrt(2 / (fan_in + fan_out))
            elif init_type == "kaiming":
                scale = math.sqrt(2 / fan_in)
            else:
                scale = 1.0

            W = torch.randn(fan_out, fan_in) * scale
            weights.append(W)

        # THEN generate input
        x = torch.randn(1, input_dim)

        std_act = []

        for W in weights:
            x = x @ W.T
            x = torch.relu(x)
            std_act.append(round(x.std().item(), 2))

        return std_act
        # The code below shows the exact right concepts but this problem is bugged with seeding
        # for i in range(num_layers):

        #     if i == 0:
        #         input_dim = input_dim
        #     else:
        #         input_dim = hidden_dim
            
        #     output_dim = hidden_dim

        #     if init_type == "xavier_init":
        #         W = torch.randn(input_dim, output_dim) * math.sqrt(2/(input_dim + output_dim))
            
        #     else:
        #         W = torch.randn(input_dim, output_dim) * math.sqrt(2/input_dim)
            
        #     z = W.T @ x
        #     x = torch.relu(z)
        #     std_act.append(round(x.std().item(), 2))

        # return std_act

