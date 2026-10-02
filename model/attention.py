import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.attention_dim = attention_dim
        self.keys = nn.Linear(embedding_dim, attention_dim, False)
        self.queries = nn.Linear(embedding_dim, attention_dim, False)
        self.values = nn.Linear(embedding_dim, attention_dim, False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, _ = x.shape
        k = self.keys(x)        # [B, T, D]
        q = self.queries(x)     # [B, T, D]
        v = self.values(x)      # [B, T, D]

        k_t = k.transpose(1, -1)
        scores = (q @ k_t) / (self.attention_dim**0.5)  # [B, T, T]

        mask = torch.tril(torch.ones(T, T))    # [T, T]
        masked_scores = scores.masked_fill(mask==0, -torch.inf)   

        attention = torch.softmax(masked_scores, dim=2)

        context_vecs = attention @ v

        return torch.round(context_vecs, decimals=4)
        