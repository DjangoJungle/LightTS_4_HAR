# models/autoformer.py

import torch
import torch.nn as nn

class SeriesDecompositionBlock(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecompositionBlock, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        # x: [B, T, C]
        x = x.permute(0, 2, 1)  # [B, C, T]
        moving_mean = self.moving_avg(x)
        moving_mean = moving_mean.permute(0, 2, 1)  # [B, T, C]
        res = x.permute(0, 2, 1) - moving_mean
        return res, moving_mean
