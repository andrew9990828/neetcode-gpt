import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.embeddings = nn.Embedding(vocabulary_size, 16)     # Assign 16 emb_dims
                                                                # to each vocab word
        self.linear = nn.Linear(16, 1)  # Input 16 embeds -> 1 output
        self.sigmoid = nn.Sigmoid()     # Squish between 0 and 1
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # x.shape -> B, T
        emb_dim = self.embeddings(x)    # [B, T, D]
        emb_dim = torch.mean(emb_dim, dim=1)    # [B]
        z = self.linear(emb_dim)    
        out = self.sigmoid(z)
        return torch.round(out, decimals=4)
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places

