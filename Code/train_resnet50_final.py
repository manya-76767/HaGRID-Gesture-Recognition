import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tqdm import tqdm

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

TRAIN_DIR = "Dataset_Split/train"
VAL_DIR = "Dataset_Split/val"
TEST_DIR = "Dataset_Split/test"

MODEL_PATH = "Models/resnet50_best.pth"

BATCH_SIZE = 32
EPOCHS = 10
LR = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)

# --------------------------------------------------
# TRANSFORMS
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# --------------------------------------------------
# DATASETS
# --------------------------------------------------

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

classes = train_dataset.classes

print("Classes:", classes)
print("Number of classes:", len(classes))

# --------------------------------------------------
# MODEL
# --------------------------------------------------

weights = models.ResNet50_Weights.DEFAULT

model = models.resnet50(weights=weights)

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LR
)

# --------------------------------------------------
# TRAINING
# --------------------------------------------------

train_acc_history = []
val_acc_history = []

train_loss_history = []
val_loss_history = []

best_val_acc = 0

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

    train_acc = 100 * correct / total

    train_loss /= len(train_loader)

    # Validation

    model.eval()

    val_loss = 0

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

    val_acc = 100 * correct / total

    val_loss /= len(val_loader)

    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Acc={train_acc:.2f}% "
        f"Val Acc={val_acc:.2f}%"
    )

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

print("\nBest Validation Accuracy:",
      round(best_val_acc,2))

# --------------------------------------------------
# TEST EVALUATION
# --------------------------------------------------

model.load_state_dict(
    torch.load(MODEL_PATH)
)

model.eval()

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        _, predicted = outputs.max(1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())

acc = accuracy_score(y_true,y_pred)
prec = precision_score(
    y_true,y_pred,
    average="weighted"
)

rec = recall_score(
    y_true,y_pred,
    average="weighted"
)

f1 = f1_score(
    y_true,y_pred,
    average="weighted"
)

print("\nTEST RESULTS")
print("Accuracy :",acc)
print("Precision:",prec)
print("Recall   :",rec)
print("F1 Score :",f1)

# --------------------------------------------------
# SAVE METRICS
# --------------------------------------------------

os.makedirs("Results",exist_ok=True)

with open(
    "Results/resnet50_metrics.txt",
    "w"
) as f:

    f.write(f"Accuracy: {acc}\n")
    f.write(f"Precision: {prec}\n")
    f.write(f"Recall: {rec}\n")
    f.write(f"F1 Score: {f1}\n")

# --------------------------------------------------
# ACCURACY GRAPH
# --------------------------------------------------

os.makedirs("Figures",exist_ok=True)

plt.figure(figsize=(8,5))

plt.plot(train_acc_history,label="Train")
plt.plot(val_acc_history,label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.savefig(
    "Figures/resnet50_accuracy.png"
)

# --------------------------------------------------
# LOSS GRAPH
# --------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(train_loss_history,label="Train")
plt.plot(val_loss_history,label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.savefig(
    "Figures/resnet50_loss.png"
)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(14,14))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

disp.plot(
    xticks_rotation=90
)

plt.savefig(
    "Figures/resnet50_confusion_matrix.png"
)

print("\nEverything saved successfully.")