# models/revin.py

import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.affine = affine
        self.subtract_last = subtract_last
        self.eps = 1e-5

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True)
            x = (x - self.mean) / (self.std + self.eps)
            if self.affine:
                x = x * self.gamma + self.beta
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)
            x = x * (self.std + self.eps) + self.mean
            return x
        else:
            raise ValueError("Mode must be 'norm' or 'denorm'")
