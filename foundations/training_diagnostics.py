import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        cur = x
        res = []

        with torch.no_grad():
            for layer in model:

                cur = layer(cur)

                if isinstance(layer, nn.Linear):

                    layers = {}
                   
                    mean = cur.mean()
                    std = cur.std()
                    dead_fraction = (cur <= 0).all(dim=0).float().mean()

                    layers["mean"] = round(mean.item(), 4)
                    layers["std"] = round(std.item(), 4)
                    layers["dead_fraction"] = round(dead_fraction.item(), 4)

                    res.append(layers)

        return res
 
    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        
        # Clear old gradients
        loss_function = nn.MSELoss()
        model.zero_grad()
        pred = model(x)
        loss = loss_function(pred, y)
        loss.backward()
        res = []

        for layer in model:
            if isinstance(layer, nn.Linear):
                stats = {}
                grad = layer.weight.grad

                mean = grad.mean()
                std = grad.std()
                norm = grad.norm()

                stats["mean"] = round(mean.item(), 4)
                stats["std"] = round(std.item(), 4)
                stats["norm"] = round(norm.item(), 4)

                res.append(stats)

        return res


    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        n = len(activation_stats)

        for i in range(n):
            if activation_stats[i]["dead_fraction"] > 0.5:
                return "dead_neurons"
            
            if gradient_stats[i]["norm"] > 1000:
                return "exploding_gradients"
            
            if gradient_stats[i]["norm"] < 1e-5:
                return "vanishing_gradients"
            
            if activation_stats[i]["std"] < 0.1:
                return "vanishing_gradients"
            
            if activation_stats[i]["std"] > 10.0:
                return "exploding_gradients"

        return "healthy"



