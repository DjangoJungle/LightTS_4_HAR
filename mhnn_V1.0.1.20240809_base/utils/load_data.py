import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pywt


def load_inertial_signals(data_dir, dataset='train'):
    signal_types = ['body_acc_x', 'body_acc_y', 'body_acc_z',
                    'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
                    'total_acc_x', 'total_acc_y', 'total_acc_z']

    signals = []
    for signal_type in signal_types:
        filename = os.path.join(data_dir, dataset, 'Inertial Signals', signal_type + '_' + dataset + '.txt')
        signal_data = np.loadtxt(filename).reshape(-1, 128)
        signals.append(signal_data)

    return np.transpose(np.array(signals), (1, 0, 2))  # Shape (n_samples, n_channels, n_timesteps)


def wavelet_decompose(signal, wavelet='haar', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs_flattened = np.concatenate(coeffs)  # Flatten the coefficients into a single vector
    return coeffs_flattened


def wavelet_decompose_signals(signals, wavelet='haar', level=3):
    n_samples, n_channels, n_timesteps = signals.shape
    # Determine the size of the flattened coefficients (this depends on the wavelet and level)
    example_decomposed = wavelet_decompose(signals[0, 0], wavelet, level)
    flattened_size = example_decomposed.shape[0]

    decomposed_signals = np.zeros((n_samples, n_channels, flattened_size))

    for i in range(n_samples):
        for j in range(n_channels):
            decomposed_signals[i, j] = wavelet_decompose(signals[i, j], wavelet, level)

    # Reshape to (n_samples, n_channels * flattened_size) for input to fully connected layers or CNN
    return decomposed_signals.reshape(n_samples, -1)


def create_dataloaders(data_dir, batch_size=64):
    X_train = load_inertial_signals(data_dir, 'train')
    X_test = load_inertial_signals(data_dir, 'test')

    # Apply wavelet decomposition to the signals
    X_train_decomposed = wavelet_decompose_signals(X_train)
    X_test_decomposed = wavelet_decompose_signals(X_test)

    y_train = np.loadtxt(os.path.join(data_dir, 'train', 'y_train.txt')) - 1
    y_test = np.loadtxt(os.path.join(data_dir, 'test', 'y_test.txt')) - 1

    # Convert decomposed signals to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_decomposed, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_decomposed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
