import torch
import torch.optim as optim
from V1.models.mhnn import MHNN
from V1.utils.load_data import create_dataloaders

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, test_loader = create_dataloaders('../data/UCI-HAR-Dataset')

# Determine input size from the preprocessed data
input_size = train_loader.dataset[0][0].shape[0]  # 获取数据的特征向量长度

# Initialize model, loss function, and optimizer
model = MHNN(num_classes=6, input_size=input_size).to(device)
criterion = torch.nn.CrossEntropyLoss()
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

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # Compare predictions with labels
        incorrect_indices = (predicted != labels)

        if incorrect_indices.any():
            # Get the incorrectly predicted labels and the actual labels
            incorrect_predictions = predicted[incorrect_indices]
            actual_labels = labels[incorrect_indices]
            # Print the incorrect predictions and their corresponding actual labels
            for pred, actual in zip(incorrect_predictions, actual_labels):
                print(f"Incorrect Prediction: {pred.item()}, Actual Label: {actual.item()}")

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Print the final accuracy
print(f'Accuracy: {100 * correct / total}%')

# Save the model
torch.save(model.state_dict(), 'mhnn_model.pth')
