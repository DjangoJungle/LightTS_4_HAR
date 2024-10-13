# models/transformer/embedding.py

import torch
import torch.nn as nn

class DataEmbedding(nn.Module):
    def __init__(self, input_dim, model_dim, embed_type='fixed', freq='h', dropout=0.1, with_pos=False):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(input_dim, model_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, input_dim, model_dim)) if with_pos else None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        # x: [B, T, D]
        x = self.value_embedding(x)
        if self.position_embedding is not None:
            x = x + self.position_embedding
        return self.dropout(x)
