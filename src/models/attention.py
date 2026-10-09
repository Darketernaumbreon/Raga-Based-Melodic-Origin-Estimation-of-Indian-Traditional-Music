# Write your TemporalAttention class here
import torch
import torch.nn as nn 


class Attention(nn.Module):
    def __init__(self,  hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        scores = self.attention(x)
        weights = torch.softmax(scores,dim = 1)
        weighted_features = x * weights
        context_vector = weighted_features.sum(dim=1)
        return context_vector
        
