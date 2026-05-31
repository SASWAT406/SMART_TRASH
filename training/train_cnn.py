from pathlib import Path
from functools import lru_cache
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import random_split

from cnn import GarbageCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["metal", "paper", "plastic"]
DATASET_PATH = (
    PROJECT_ROOT / "dataset_cropped"
    if (PROJECT_ROOT / "dataset_cropped").exists()
    else PROJECT_ROOT / "dataset"
)
MODEL_PATH = PROJECT_ROOT / "models" / "garbage_cnn.pth"
BATCH_SIZE = 64
EPOCHS = 18
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
PATIENCE = 4
PERSONAL_SAMPLE_REPEAT = 35
IMAGE_SIZE = 128
SEED = 42


class GarbageImageFolder(datasets.ImageFolder):

    def __init__(self, *args, personal_only=False, **kwargs):
        self.personal_only = personal_only
        super().__init__(*args, **kwargs)

        original_count = len(self.samples)
        self.samples = [
            sample
            for sample in self.samples
            if self.keep_sample(sample)
        ]
        self.imgs = self.samples
        self.targets = [label for _, label in self.samples]
        self.filtered_count = original_count - len(self.samples)

    def keep_sample(self, sample):
        image_path = Path(sample[0])
        is_personal = image_path.name.startswith("personal_")

        if self.personal_only:
            return is_personal

        if is_personal:
            return True

        return image_is_usable(image_path)

    def find_classes(self, directory):

        classes = CLASS_NAMES
        class_to_idx = {
            class_name: index
            for index, class_name in enumerate(classes)
        }

        return classes, class_to_idx


@lru_cache(maxsize=None)
def image_is_usable(image_path):
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if image.width < 24 or image.height < 24:
                return False

            image.thumbnail((96, 96))
            pixels = np.asarray(image)
    except OSError:
        return False

    gray = pixels.mean(axis=2)
    black_fraction = float((gray < 8).mean())
    color_variation = float((pixels.max(axis=2) - pixels.min(axis=2)).mean())

    if black_fraction > 0.18:
        return False

    if color_variation < 5 and black_fraction > 0.04:
        return False

    return True


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(dataset, indices):
    counts = torch.zeros(len(CLASS_NAMES), dtype=torch.float)

    for index in indices:
        _, label = dataset.samples[index]
        counts[label] += 1

    counts = torch.clamp(counts, min=1)
    weights = counts.sum() / (len(CLASS_NAMES) * counts)

    return weights


def main():
    seed_everything(SEED)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.7, 1.0),
            ratio=(0.8, 1.25),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.25,
            hue=0.04,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        ),
        transforms.RandomErasing(
            p=0.2,
            scale=(0.02, 0.12),
            ratio=(0.3, 3.3),
            value="random",
        ),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 16),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        ),
    ])

    train_dataset = GarbageImageFolder(
        root=DATASET_PATH,
        transform=train_transform,
    )

    valid_dataset = GarbageImageFolder(
        root=DATASET_PATH,
        transform=valid_transform,
    )

    personal_indices = [
        index
        for index, sample in enumerate(train_dataset.samples)
        if Path(sample[0]).name.startswith("personal_")
    ]
    regular_indices = [
        index
        for index in range(len(train_dataset))
        if index not in personal_indices
    ]

    valid_size = int(len(regular_indices) * VALIDATION_SPLIT)
    train_size = len(regular_indices) - valid_size

    train_subset, valid_subset = random_split(
        regular_indices,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_indices = list(train_subset) + personal_indices * PERSONAL_SAMPLE_REPEAT
    valid_indices = list(valid_subset)

    train_data = Subset(train_dataset, train_indices)
    valid_data = Subset(valid_dataset, valid_indices)

    if not train_indices or not valid_indices:
        raise RuntimeError("Dataset split is empty. Check the dataset folders.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    personal_dataset = None
    personal_root = PROJECT_ROOT / "dataset"

    if personal_root.exists() and personal_root != DATASET_PATH:
        personal_dataset = GarbageImageFolder(
            root=personal_root,
            transform=train_transform,
            personal_only=True,
        )

        if len(personal_dataset) > 0:
            personal_data = Subset(
                personal_dataset,
                list(range(len(personal_dataset))) * PERSONAL_SAMPLE_REPEAT,
            )
            train_data = ConcatDataset([train_data, personal_data])

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=pin_memory,
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory,
    )

    model = GarbageCNN(num_classes=len(CLASS_NAMES)).to(device)
    weights = class_weights(train_dataset, train_indices).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    print(f"Classes: {train_dataset.classes}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Filtered images: {train_dataset.filtered_count}")
    if personal_dataset is not None:
        print(f"Personal images: {len(personal_dataset)}")
    print(f"Training images: {len(train_data)}")
    print(f"Validation images: {len(valid_data)}")
    print(f"Device: {device}")
    print(f"Class weights: {[round(weight.item(), 3) for weight in weights]}")

    epochs = EPOCHS
    best_valid_loss = float("inf")
    best_valid_accuracy = 0.0
    patience_counter = 0

    for epoch in range(epochs):
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

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()

        valid_loss = valid_loss / len(valid_loader)
        valid_accuracy = 100 * valid_correct / valid_total

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Accuracy: {train_accuracy:.2f}% "
            f"Valid Loss: {valid_loss:.4f} "
            f"Valid Accuracy: {valid_accuracy:.2f}%"
        )

        if (
            valid_accuracy > best_valid_accuracy
            or (
                valid_accuracy == best_valid_accuracy
                and valid_loss < best_valid_loss
            )
        ):
            best_valid_loss = valid_loss
            best_valid_accuracy = valid_accuracy
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE,
                    "dataset_path": str(DATASET_PATH),
                },
                MODEL_PATH
            )

            print("Best model saved.")
        else:
            patience_counter += 1

            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

        scheduler.step(valid_loss)

    print("Model Saved!")


if __name__ == "__main__":
    main()
