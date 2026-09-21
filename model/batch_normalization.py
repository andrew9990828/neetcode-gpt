import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)
        beta = np.array(beta, dtype=float)
        running_mean = np.array(running_mean, dtype=float)
        running_var = np.array(running_var, dtype=float)

        if training:
            mean_batch = np.mean(x, axis=0)
            var = np.mean((x - mean_batch) ** 2, axis=0)

            running_mean = (1 - momentum) * running_mean + (momentum * mean_batch)
            running_var = (1 - momentum) * running_var + (momentum * var)
        else:
            mean_batch = running_mean
            var = running_var
        
        x_hat = ((x - mean_batch)/np.sqrt(var + eps))
        y = gamma * x_hat + beta
        return (np.round(y, 4), np.round(running_mean, 4), np.round(running_var, 4))
        