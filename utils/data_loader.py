import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(data_path):
    train_data = pd.read_csv(f'{data_path}/train/X_train.txt', delim_whitespace=True, header=None)
    train_labels = pd.read_csv(f'{data_path}/train/y_train.txt', delim_whitespace=True, header=None)
    test_data = pd.read_csv(f'{data_path}/test/X_test.txt', delim_whitespace=True, header=None)
    test_labels = pd.read_csv(f'{data_path}/test/y_test.txt', delim_whitespace=True, header=None)

    # 转换为Tensor并调整形状
    train_data = torch.tensor(train_data.values, dtype=torch.float32).view(-1, 1, 561)
    train_labels = torch.tensor(train_labels.values, dtype=torch.long).squeeze() - 1
    test_data = torch.tensor(test_data.values, dtype=torch.float32).view(-1, 1, 561)
    test_labels = torch.tensor(test_labels.values, dtype=torch.long).squeeze() - 1

    # 创建DataLoader
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader
