import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load UCI-HAR dataset
def load_UCI_HAR_data():
    # Paths to the dataset files
    DATA_PATH = 'UCI HAR Dataset/'
    TRAIN = 'train/'
    TEST = 'test/'

    # Load features
    features = pd.read_csv(DATA_PATH + 'features.txt', delim_whitespace=True, header=None)
    feature_names = features.iloc[:, 1].values

    # Load training data
    X_train = pd.read_csv(DATA_PATH + TRAIN + 'X_train.txt', delim_whitespace=True, header=None)
    y_train = pd.read_csv(DATA_PATH + TRAIN + 'y_train.txt', delim_whitespace=True, header=None)
    subject_train = pd.read_csv(DATA_PATH + TRAIN + 'subject_train.txt', delim_whitespace=True, header=None)

    # Load test data
    X_test = pd.read_csv(DATA_PATH + TEST + 'X_test.txt', delim_whitespace=True, header=None)
    y_test = pd.read_csv(DATA_PATH + TEST + 'y_test.txt', delim_whitespace=True, header=None)
    subject_test = pd.read_csv(DATA_PATH + TEST + 'subject_test.txt', delim_whitespace=True, header=None)

    # Combine train and test data
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test]).values.ravel()

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

# Prepare the dataset for PyTorch
class HAR_Dataset(Dataset):
    def __init__(self, X, y, sequence_length=128):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length

        # Reshape data into sequences
        self.X = self.X.reshape(-1, sequence_length, self.X.shape[1])
        self.y = self.y[::sequence_length]

        # Encode labels
        label_encoder = LabelEncoder()
        self.y = label_encoder.fit_transform(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), self.y[idx]

# Multilevel Discrete Wavelet Decomposition
def multilevel_wavelet_decomposition(x, wavelet='haar', level=3):
    coeffs = []
    for i in range(x.shape[1]):  # For each channel
        c = x[:, i]
        cA = c
        cD_list = []
        for l in range(level):
            cA, cD = pywt.dwt(cA, wavelet)
            cD_list.append(cD)
        coeffs.append((cA, cD_list))
    return coeffs

# Model Definition
class MHNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(MHNN, self).__init__()
        # Heterogeneous Feature Learner

        # For original signals
        self.original_conv = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # For H1
        self.h1_conv = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # For H2
        self.h2_conv = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # For H3
        self.h3_mlp = nn.Sequential(
            nn.Linear(input_channels, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
        )

        # Cross Aggregation Module
        self.cross_conv = nn.Sequential(
            nn.Conv1d(128 * 3, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Classifier
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_length, channels)
        x = x.permute(0, 2, 1)  # (batch_size, channels, seq_length)

        # Multilevel Discrete Wavelet Decomposition
        # For simplicity, we will simulate MDWD by downsampling
        H1 = F.avg_pool1d(x, kernel_size=2)
        H2 = F.avg_pool1d(H1, kernel_size=2)
        H3 = F.avg_pool1d(H2, kernel_size=2)

        # Heterogeneous Feature Learner
        original_feat = self.original_conv(x)

        h1_feat = self.h1_conv(H1)

        h2_feat = self.h2_conv(H2)

        # Flatten H3 for MLP
        h3_feat = H3.view(H3.size(0), -1)
        h3_feat = self.h3_mlp(h3_feat)

        # Cross Aggregation Module
        # For simplicity, we will concatenate features
        # and pass through cross_conv
        # We need to make sure all features have the same shape
        # So we will adapt dimensions as necessary

        # Resize h3_feat to match other feature dimensions
        h3_feat = h3_feat.unsqueeze(2)  # (batch_size, 128, 1)

        # Adjust features to have same temporal dimension
        min_length = min(original_feat.size(2), h1_feat.size(2), h2_feat.size(2))
        original_feat = original_feat[:, :, :min_length]
        h1_feat = h1_feat[:, :, :min_length]
        h2_feat = h2_feat[:, :, :min_length]
        h3_feat = h3_feat.expand(-1, -1, min_length)

        combined_feat = torch.cat([original_feat, h1_feat, h2_feat], dim=1)

        cross_feat = self.cross_conv(combined_feat)

        # Global average pooling
        pooled = F.adaptive_avg_pool1d(cross_feat, 1).squeeze(2)

        # Classifier
        out = self.classifier(pooled)

        return out

# Training and Evaluation Functions
def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print('Epoch {}/{} Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print('Accuracy: {:.4f} Precision: {:.4f} Recall: {:.4f} F1 Score: {:.4f}'.format(
        accuracy, precision, recall, f1))

    return accuracy, precision, recall, f1

# Main Execution
if __name__ == '__main__':
    # Load data
    X, y = load_UCI_HAR_data()

    # Split data into train and test sets
    split_ratio = 0.8
    split_index = int(len(y) * split_ratio)
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]

    # Create datasets
    sequence_length = 128
    train_dataset = HAR_Dataset(X_train, y_train, sequence_length=sequence_length)
    test_dataset = HAR_Dataset(X_test, y_test, sequence_length=sequence_length)

    # Create dataloaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, criterion, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_channels = train_dataset.X.shape[2]
    num_classes = len(np.unique(y))

    model = MHNN(input_channels=input_channels, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Train the model
    num_epochs = 25
    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

    # Evaluate the model
    evaluate_model(model, test_loader)
