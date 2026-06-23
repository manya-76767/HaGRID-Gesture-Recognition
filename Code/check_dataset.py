from torchvision import datasets

train_dataset = datasets.ImageFolder("Dataset_Split/train")

print("Number of classes:", len(train_dataset.classes))
print("Classes:")

for i, c in enumerate(train_dataset.classes):
    print(i, c)

print("\nTotal training images:", len(train_dataset))