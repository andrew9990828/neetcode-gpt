import numpy as np
from typing import List


class Solution:
    def MSE_loss(self, predictions, y_true):
        return np.mean((predictions - y_true)**2)
    
    def reLu(self, forward):
        return np.maximum(0, forward)

    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        # convert regular lists to numPy arrays
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
    
        # Forward
        ans = {}
        z1 = W1 @ x + b1
        a1 = self.reLu(z1)
        z2 = W2 @ a1 + b2
        preds = self.reLu(z2)
        loss = self.MSE_loss(preds, y_true)

        # Backward
        dz2 = (2 / len(y_true)) * (z2 - y_true)
        dW2 = np.outer(dz2, a1)
        db2 = dz2
        da1 = W2.T @ dz2
        dz1 = da1 * (z1 > 0)
        dW1 = np.outer(dz1, x)
        db1 = dz1
        

        ans["loss"] = np.round(loss, 5)
        ans["dW1"] = np.round(dW1, 5)
        ans["db1"] = np.round(db1, 5)
        ans["dW2"] = np.round(dW2, 5)
        ans["db2"] = np.round(db2, 5)

        return ans