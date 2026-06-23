import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Parameters
# -----------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 2
LEARNING_RATE = 0.0001

# -----------------------------
# Image Transformations
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# Datasets
# -----------------------------
train_dataset = datasets.ImageFolder(
    "Dataset_Split/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    "Dataset_Split/val",
    transform=val_transform
)

print("Classes:", train_dataset.classes)
print("Number of classes:", len(train_dataset.classes))

# -----------------------------
# DataLoaders
# -----------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# -----------------------------
# Model
# -----------------------------
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(train_dataset.classes)
)

model = model.to(device)

# -----------------------------
# Loss + Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

    train_acc = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Loss: {running_loss:.4f} "
        f"Train Acc: {train_acc:.2f}%"
    )

print("Training Finished")

torch.save(
    model.state_dict(),
    "Models/resnet50_phase1.pth"
)

print("Model saved.")