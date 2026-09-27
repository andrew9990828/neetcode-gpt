import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        dead_fractions = []
        
        for layer in model:
            x = layer(x)

            if isinstance(layer, nn.ReLU):
                relu = layer(x)
                count_zeros = (relu <= 0).all(dim=0).float().mean()
                dead_fractions.append(torch.round(count_zeros, decimals=4))
        
        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        prev_frac = None
        n = len(dead_fractions) - 1
        largest_frac = max(dead_fractions)

        for i, fraction in enumerate(dead_fractions):
            if fraction > 0.5:
                return "use_leaky_relu"
            if i == 0 and fraction > 0.3:
                return "reinitialize"
            if (prev_frac and prev_frac < fraction and i == n and 
                fraction > 0.1):
                return "reduce_learning_rate"
            if largest_frac < 0.1:
                return "healthy"
            prev_frac = fraction
        
        return "healthy"
            
