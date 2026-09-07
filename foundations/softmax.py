import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        largest = np.max(z)

        for i, val in enumerate(z):
            z[i] = np.exp(val - largest)
        
        ans = np.array([], dtype=np.float64)
        summation = np.sum(z)

        for val in z:
            ans = np.append(ans, val/summation, axis=None)

        return np.round(ans, 4)
