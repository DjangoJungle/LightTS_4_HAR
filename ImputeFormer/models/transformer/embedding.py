import torch.nn as nn

class DataEmbedding(nn.Module):
    def __init__(self, model_dim, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(1, model_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [batch_size, n_steps, n_features, 1]
        x = self.value_embedding(x)  # [batch_size, n_steps, n_features, model_dim]
        return self.dropout(x)
