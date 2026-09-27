import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.input_dim = 784
        self.embed_dim = 512
        self.output_dim = 10
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.layer1 = nn.Linear(self.input_dim, self.embed_dim)
        self.layer2 = nn.Linear(self.embed_dim, self.output_dim)

        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        x = images
        x = self.layer1(x)
        a = self.relu(x)
        z = self.dropout(a)
        x = self.layer2(z)
        out = self.sigmoid(x) 
        return torch.round(out, decimals=4)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        
