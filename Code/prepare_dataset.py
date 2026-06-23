import os
import shutil
import random
from pathlib import Path

random.seed(42)

SOURCE_DIR = Path("Dataset/hagrid_30k")
DEST_DIR = Path("Dataset_Split")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DEST_DIR.mkdir(exist_ok=True)

for gesture_folder in SOURCE_DIR.iterdir():

    if not gesture_folder.is_dir():
        continue

    class_name = gesture_folder.name.replace("train_val_", "")

    images = list(gesture_folder.glob("*"))

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = int(total * (TRAIN_RATIO + VAL_RATIO))

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for split_name, split_images in [
        ("train", train_imgs),
        ("val", val_imgs),
        ("test", test_imgs)
    ]:

        split_class_dir = DEST_DIR / split_name / class_name
        split_class_dir.mkdir(parents=True, exist_ok=True)

        for img in split_images:
            shutil.copy2(img, split_class_dir / img.name)

    print(f"{class_name} completed")

print("Dataset splitting complete.")