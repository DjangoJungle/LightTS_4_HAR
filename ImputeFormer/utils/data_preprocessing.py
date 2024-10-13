import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(data_dir):
    # 定义传感器数据文件名
    INPUT_SIGNAL_TYPES = [
        'body_acc_x',
        'body_acc_y',
        'body_acc_z',
        'body_gyro_x',
        'body_gyro_y',
        'body_gyro_z',
        'total_acc_x',
        'total_acc_y',
        'total_acc_z',
    ]

    def load_X(X_signals_paths):
        X_signals = []
        for signal_type_path in X_signals_paths:
            X_signal = np.loadtxt(signal_type_path)
            X_signals.append(X_signal)
        # 转置并交换轴，使形状为 [样本数, 时间步, 特征数]
        return np.transpose(np.array(X_signals), (1, 2, 0))

    def load_y(y_path):
        y = np.loadtxt(y_path).astype(int) - 1  # 标签从0开始
        return y

    # 加载训练数据
    X_train_signals_paths = [
        os.path.join(data_dir, 'train', 'Inertial Signals', signal + '_train.txt') for signal in INPUT_SIGNAL_TYPES
    ]
    X_train = load_X(X_train_signals_paths)
    y_train = load_y(os.path.join(data_dir, 'train', 'y_train.txt'))

    # 加载测试数据
    X_test_signals_paths = [
        os.path.join(data_dir, 'test', 'Inertial Signals', signal + '_test.txt') for signal in INPUT_SIGNAL_TYPES
    ]
    X_test = load_X(X_test_signals_paths)
    y_test = load_y(os.path.join(data_dir, 'test', 'y_test.txt'))

    # 将数据合并
    X = np.vstack((X_train, X_test))
    y = np.hstack((y_train, y_test))

    return X, y


def preprocess_data(X, y):
    # 对每个特征独立进行标准化
    n_samples, n_steps, n_features = X.shape
    X = X.reshape(n_samples * n_steps, n_features)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(n_samples, n_steps, n_features)
    return X, y


def get_data_loaders(X, y, batch_size=64, test_size=0.2, random_state=42):
    from torch.utils.data import TensorDataset, DataLoader
    import torch

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 转换为张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # 创建数据集和数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
