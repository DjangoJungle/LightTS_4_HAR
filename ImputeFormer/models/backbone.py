import torch
import torch.nn as nn
from .attention import ProjectedAttentionLayer, EmbeddedAttentionLayer
from .mlp import MLP
from .transformer.embedding import DataEmbedding


class HARModel(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        n_classes,
        d_model=64,
        n_heads=8,
        n_layers=2,
        dim_proj=16,
        node_embedding_dim=16,
        feed_forward_dim=128,
        dropout=0.1,
    ):
        super(HARModel, self).__init__()

        self.embedding = DataEmbedding(d_model, dropout)

        # Node embedding for spatial attention
        self.node_embedding = nn.Parameter(torch.randn(n_features, node_embedding_dim))

        # Temporal attention layers
        self.temporal_layers = nn.ModuleList([
            ProjectedAttentionLayer(
                seq_len=n_steps,
                dim_proj=dim_proj,
                d_model=d_model,
                n_heads=n_heads,
                d_ff=feed_forward_dim,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Spatial attention layers
        self.spatial_layers = nn.ModuleList([
            EmbeddedAttentionLayer(
                model_dim=d_model,
                node_embedding_dim=node_embedding_dim,
                feed_forward_dim=feed_forward_dim,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        # x: [batch_size, n_steps, n_features]
        x = x.unsqueeze(-1)  # [batch_size, n_steps, n_features, 1]
        x = self.embedding(x)  # [batch_size, n_steps, n_features, d_model]

        for temp_layer, spatial_layer in zip(self.temporal_layers, self.spatial_layers):
            # Temporal attention
            x = temp_layer(x)
            # Spatial attention
            x = spatial_layer(x, self.node_embedding)

        # Pooling over time and nodes
        x = x.mean(dim=[1, 2])  # [batch_size, d_model]
        out = self.classifier(x)  # [batch_size, n_classes]
        return out