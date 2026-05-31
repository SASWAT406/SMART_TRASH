from pathlib import Path
import sys

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "inference"))

from predict import predict


CLASSES = ["metal", "paper", "plastic"]
DATASET_PATH = (
    PROJECT_ROOT / "dataset_cropped"
    if (PROJECT_ROOT / "dataset_cropped").exists()
    else PROJECT_ROOT / "dataset"
)
SAMPLE_LIMIT = 200


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


def main():
    total_correct = 0
    total_images = 0

    print(f"Dataset: {DATASET_PATH}")

    for class_name in CLASSES:
        image_paths = [
            image_path
            for image_path in sorted((DATASET_PATH / class_name).glob("*.*"))
            if image_is_usable(image_path)
        ][:SAMPLE_LIMIT]
        correct = 0

        for image_path in image_paths:
            result = predict(image_path)

            if result["class_name"] == class_name:
                correct += 1

        total_correct += correct
        total_images += len(image_paths)

        print(f"{class_name}: {correct}/{len(image_paths)} correct")

    accuracy = 100 * total_correct / max(total_images, 1)
    print(f"Overall: {total_correct}/{total_images} correct ({accuracy:.2f}%)")


if __name__ == "__main__":
    main()
