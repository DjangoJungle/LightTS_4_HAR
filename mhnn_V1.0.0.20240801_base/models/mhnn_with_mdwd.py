import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.mdwd import mdwd


class MHNNWithMDWD(nn.Module):
    def __init__(self):
        super(MHNNWithMDWD, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # 更新全连接层输入大小
        self.fc1 = nn.Linear(256 * 70, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        batch_size = x.shape[0]
        mdwd_coeffs = []

        # 对每个样本进行MDWD处理
        for i in range(batch_size):
            sample = x[i].squeeze().numpy()
            coeffs = mdwd(sample)
            coeffs_flat = np.hstack([c.flatten() for c in coeffs])
            mdwd_coeffs.append(coeffs_flat)

        x = torch.tensor(mdwd_coeffs, dtype=torch.float32)
        x = x.view(batch_size, 1, -1)

        # 通过卷积层
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        # 通过全连接层
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
