import torch
import torch.nn as nn


class HeterogeneousFeatureLearner(nn.Module):
    def __init__(self, input_size):
        super(HeterogeneousFeatureLearner, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.cnn1 = nn.Conv1d(1, 128, kernel_size=7)
        self.cnn2 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=7),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3),
            nn.ReLU()
        )

    def forward(self, x):
        # Reshape input to match CNN requirements
        x = x.unsqueeze(1)  # Add a channel dimension for CNN layers: shape becomes (batch_size, 1, input_size)
        x1 = self.fc1(x.squeeze(1))  # Fully connected layers expect flattened input
        x2 = self.cnn1(x)  # CNN expects input in (batch_size, channels, sequence_length)
        x2 = torch.mean(x2, dim=2)  # Global Average Pooling to reduce the dimensions to 2D
        x3 = self.cnn2(x)
        x3 = torch.mean(x3, dim=2)  # Global Average Pooling to reduce the dimensions to 2D
        return x1, x2, x3


class CrossAggregationModule(nn.Module):
    def __init__(self):
        super(CrossAggregationModule, self).__init__()
        self.fc = nn.Linear(128 * 3, 128)  # Adjusted to handle concatenated 2D features

    def forward(self, x1, x2, x3):
        # Ensure all inputs are 2D before concatenation
        combined = torch.cat([x1, x2, x3], dim=1)
        return self.fc(combined)


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        return nn.functional.softmax(self.fc(x), dim=1)


class MHNN(nn.Module):
    def __init__(self, num_classes, input_size):
        super(MHNN, self).__init__()
        self.learner = HeterogeneousFeatureLearner(input_size)
        self.aggregator = CrossAggregationModule()
        self.classifier = Classifier(num_classes)

    def forward(self, x):
        features = self.learner(x)
        combined_features = self.aggregator(*features)
        output = self.classifier(combined_features)
        return output
