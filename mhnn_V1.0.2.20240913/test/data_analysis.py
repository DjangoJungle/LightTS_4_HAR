import torch
import torch.optim as optim
from models.mhnn import MHNN
from utils.load_data import create_dataloaders
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, test_loader = create_dataloaders('../data/UCI-HAR-Dataset')

# Determine input size from the preprocessed data
input_size = train_loader.dataset[0][0].shape[0]  # 获取数据的特征向量长度

# Initialize model, loss function, and optimizer
model = MHNN(num_classes=6, input_size=input_size).to(device)

# 定义类别权重
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=device)

# 将类别权重传递给损失函数
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=5e-4)

# Training loop
epochs = 15
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}')

# Evaluation and Confusion Matrix
model.eval()
all_labels = []
all_predictions = []

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Print the final accuracy
print(f'Accuracy: {100 * correct / total}%')

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(all_labels), yticklabels=np.unique(all_labels))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
