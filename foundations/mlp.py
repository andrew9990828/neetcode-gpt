import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def relu(self, z):
        return np.maximum(0, z)

    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        z_next = None
        A = None
        for i in range(len(weights)):
            W = weights[i]

            if z_next is None:
                z_next = x @ W + biases[i]
                A = self.relu(z_next)

            elif i < len(weights) - 1:
                z_next = A @ W + biases[i]
                A = self.relu(z_next)
            
            else:
                z_next = A @ W + biases[i] 
        
        return np.round(z_next, 5)
        
