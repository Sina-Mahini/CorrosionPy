"""
Corrosion Image Classification Pipeline:
This script sets up the training and evaluation pipeline for a corrosion image classification task.
It includes:
1. Loading the dataset and applying transformations.
2. Defining a custom CNN architecture for grayscale images.
3. Training the model on the dataset.
4. Evaluating the model's performance.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Paths
output_root = r""  # Path to the processed dataset

# Step 1: Define transformations for training and testing
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),  # Match 256x256 resized images
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Step 2: Load dataset and split into training and testing sets
dataset = datasets.ImageFolder(root=output_root, transform=train_transform)
train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Apply test transform to the test dataset
test_dataset.dataset.transform = test_transform

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# Step 3: Define the model
# Custom CNN adapted for 1-channel 512x512 grayscale input
class CorrosionCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        # Feature extractor: stack of Conv-BN-ReLU blocks with pooling
        self.features = nn.Sequential(
            # 1 x 512 x 512 -> 32 x 256 x 256
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 32 x 256 x 256 -> 64 x 128 x 128
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 64 x 128 x 128 -> 128 x 64 x 64
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 128 x 64 x 64 -> 256 x 32 x 32
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 256 x 32 x 32 -> 512 x 16 x 16, then global average pool
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # -> 512 x 1 x 1
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


num_classes = len(dataset.classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CorrosionCNN(num_classes).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

# Step 4: Training the model
epochs = 100
train_losses = []
train_accuracies = []
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        if inputs.shape[1] == 3:  # Ensure inputs are single-channel
            inputs = inputs.mean(dim=1, keepdim=True)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

# Step 5: Evaluate the model
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        if inputs.shape[1] == 3:  # Ensure inputs are single-channel
            inputs = inputs.mean(dim=1, keepdim=True)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        test_total += targets.size(0)
        test_correct += (predicted == targets).sum().item()

test_acc = test_correct / test_total
print(f"Test Accuracy: {test_acc:.4f} ({test_correct}/{test_total})")


# Step 6: Plot training loss and accuracy
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss', color='tab:blue')
ax1.plot(train_losses, color='tab:blue', linestyle='--', label='Train Loss')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Train Accuracy', color='tab:red')
ax2.plot(train_accuracies, color='tab:red', label='Train Accuracy')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Training Loss and Accuracy')
fig.tight_layout()
plt.show()