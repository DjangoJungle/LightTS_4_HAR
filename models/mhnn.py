import torch.nn as nn


class MHNN(nn.Module):
    def __init__(self):
        super(MHNN, self).__init__()
        # 定义多级信号提取器和异构特征学习器
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # 更新全连接层输入大小
        self.fc1 = nn.Linear(256 * 70, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        # 展平特征图
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
