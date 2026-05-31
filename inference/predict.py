from pathlib import Path
from functools import lru_cache
import sys

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

#venv\Scripts\python.exe inference\predict.py "test_images\image.png"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "garbage_cnn.pth"
DEFAULT_CLASS_NAMES = ["metal", "paper", "plastic"]
DEFAULT_IMAGE_SIZE = 128

sys.path.append(str(PROJECT_ROOT))

from training.cnn import GarbageCNN


def build_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        ),
    ])


def checkpoint_values(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return (
            checkpoint["model_state_dict"],
            checkpoint.get("class_names", DEFAULT_CLASS_NAMES),
            checkpoint.get("image_size", DEFAULT_IMAGE_SIZE),
        )

    return checkpoint, DEFAULT_CLASS_NAMES, DEFAULT_IMAGE_SIZE


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Train the model first with: "
            r"venv\Scripts\python.exe training\train_cnn.py"
        )

    return load_model_for_version(MODEL_PATH.stat().st_mtime_ns)


@lru_cache(maxsize=1)
def load_model_for_version(_model_version):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict, class_names, image_size = checkpoint_values(checkpoint)

    model = GarbageCNN(num_classes=len(class_names)).to(device)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise RuntimeError(
            "The saved model does not match the current CNN architecture. "
            "Retrain it with: "
            r"venv\Scripts\python.exe training\train_cnn.py"
        ) from error

    model.eval()

    return model, device, class_names, int(image_size)


def border_color(image):
    small = image.copy()
    small.thumbnail((64, 64))
    pixels = np.asarray(small, dtype=np.uint8)

    border_pixels = np.concatenate([
        pixels[0, :, :],
        pixels[-1, :, :],
        pixels[:, 0, :],
        pixels[:, -1, :],
    ])

    color = np.median(border_pixels, axis=0)

    return tuple(int(channel) for channel in color)


def square_pad(image):
    width, height = image.size

    if width == height:
        return image.copy()

    size = max(width, height)
    padded = Image.new("RGB", (size, size), border_color(image))
    padded.paste(image, ((size - width) // 2, (size - height) // 2))

    return padded


def center_square_crop(image):
    width, height = image.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2

    return image.crop((left, top, left + size, top + size))


def foreground_crop(image):
    width, height = image.size
    analysis_image = image.copy()
    analysis_image.thumbnail((256, 256))
    pixels = np.asarray(analysis_image, dtype=np.int16)
    small_height, small_width, _ = pixels.shape

    border_pixels = np.concatenate([
        pixels[0, :, :],
        pixels[-1, :, :],
        pixels[:, 0, :],
        pixels[:, -1, :],
    ])
    background = np.median(border_pixels, axis=0)
    distance = np.sqrt(np.sum((pixels - background) ** 2, axis=2))
    threshold = max(30.0, float(np.percentile(distance, 70)))
    mask = distance > threshold
    mask_area = float(mask.mean())

    if mask_area < 0.02 or mask_area > 0.88:
        return None

    rows, columns = np.where(mask)

    if len(rows) == 0 or len(columns) == 0:
        return None

    left = columns.min() / small_width * width
    right = (columns.max() + 1) / small_width * width
    top = rows.min() / small_height * height
    bottom = (rows.max() + 1) / small_height * height
    pad_x = (right - left) * 0.12
    pad_y = (bottom - top) * 0.12

    left = max(int(left - pad_x), 0)
    top = max(int(top - pad_y), 0)
    right = min(int(right + pad_x), width)
    bottom = min(int(bottom + pad_y), height)

    if right - left < 16 or bottom - top < 16:
        return None

    return image.crop((left, top, right, bottom))


def candidate_images(image):
    candidates = [
        (square_pad(image), 0.6),
        (center_square_crop(image), 1.4),
    ]

    cropped = foreground_crop(image)

    if cropped is not None:
        candidates.append((square_pad(cropped), 1.8))

    unique_candidates = []
    fingerprints = set()

    for candidate, weight in candidates:
        fingerprint = candidate.resize((16, 16)).tobytes()

        if fingerprint in fingerprints:
            continue

        fingerprints.add(fingerprint)
        unique_candidates.append((candidate, weight))

    return unique_candidates


def predict_image(image):
    model, device, class_names, image_size = load_model()
    preprocess = build_transform(image_size)

    image = ImageOps.exif_transpose(image).convert("RGB")
    candidates = candidate_images(image)
    image_tensors = [
        preprocess(candidate)
        for candidate, _weight in candidates
    ]
    weights = torch.tensor(
        [weight for _candidate, weight in candidates],
        dtype=torch.float32,
        device=device,
    )
    batch = torch.stack(image_tensors).to(device)

    with torch.no_grad():
        outputs = model(batch)
        variant_probabilities = torch.softmax(outputs, dim=1)
        probabilities = (
            variant_probabilities * weights.unsqueeze(1)
        ).sum(dim=0) / weights.sum()
        confidence, predicted_index = torch.max(probabilities, dim=0)

    class_name = class_names[predicted_index.item()]
    class_probabilities = {
        class_name: probabilities[index].item()
        for index, class_name in enumerate(class_names)
    }

    return {
        "class_name": class_name,
        "confidence": confidence.item(),
        "probabilities": class_probabilities,
        "variants": len(image_tensors),
    }


def predict(image_path):
    image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(image_path) as image:
        return predict_image(image)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: "
            r"venv\Scripts\python.exe inference\predict.py "
            r'"test_images\image.png"'
        )
        sys.exit(1)

    try:
        result = predict(sys.argv[1])
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Prediction failed: {error}")
        sys.exit(1)

    print(f"Prediction: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Image variants: {result['variants']}")
    print("Probabilities:")
    for class_name, probability in result["probabilities"].items():
        print(f"- {class_name}: {probability:.2%}")
