# models/backbone.py

import torch
import torch.nn as nn

from .layers import TimesBlock


class BackboneTimesNet(nn.Module):
    def __init__(
        self,
        n_layers,
        n_steps,
        n_pred_steps,
        top_k,
        d_model,
        d_ffn,
        n_kernels,
        n_classes,
        n_features,
    ):
        super().__init__()

        self.seq_len = n_steps
        self.n_layers = n_layers
        self.n_pred_steps = n_pred_steps
        self.top_k = top_k

        # 如果输入特征维度不等于 d_model，需要添加映射层
        self.input_projection = nn.Linear(n_features, d_model)

        self.model = nn.ModuleList(
            [TimesBlock(n_steps, n_pred_steps, top_k, d_model, d_ffn, n_kernels) for _ in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, X) -> torch.Tensor:
        # X: [B, T, n_features]
        X = self.input_projection(X)  # [B, T, d_model]

        for i in range(self.n_layers):
            X = self.layer_norm(self.model[i](X))

        X = X.permute(0, 2, 1)  # [B, d_model, T]
        X = self.pooling(X)     # [B, d_model, 1]
        X = X.squeeze(-1)       # [B, d_model]
        output = self.fc(X)     # [B, n_classes]
        return output
