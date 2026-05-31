from pathlib import Path

from PIL import Image


dataset_path = Path(
    r"C:\Users\saswa\.cache\kagglehub\datasets\alexcsl\trashnet-dataset\versions\1"
)

output_dataset = Path("dataset_cropped")

classes = {
    "2": "metal",
    "3": "paper",
    "4": "plastic",
}


for class_name in classes.values():
    (output_dataset / class_name).mkdir(parents=True, exist_ok=True)


copied_count = {
    "metal": 0,
    "paper": 0,
    "plastic": 0,
}

for split_name in ["train", "valid", "test"]:
    image_folder = dataset_path / split_name / "images"
    label_folder = dataset_path / split_name / "labels"

    if not image_folder.exists() or not label_folder.exists():
        continue

    for label_path in label_folder.glob("*.txt"):
        image_path = image_folder / label_path.with_suffix(".jpg").name

        if not image_path.exists():
            continue

        with Image.open(image_path).convert("RGB") as image:
            image_width, image_height = image.size

            for object_index, line in enumerate(label_path.read_text().splitlines()):
                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id, x_center, y_center, width, height = parts

                if class_id not in classes:
                    continue

                class_name = classes[class_id]
                x_center = float(x_center) * image_width
                y_center = float(y_center) * image_height
                width = float(width) * image_width
                height = float(height) * image_height

                left = max(int(x_center - width / 2), 0)
                top = max(int(y_center - height / 2), 0)
                right = min(int(x_center + width / 2), image_width)
                bottom = min(int(y_center + height / 2), image_height)

                if right <= left or bottom <= top:
                    continue

                cropped_image = image.crop((left, top, right, bottom))
                output_name = f"{split_name}_{label_path.stem}_{object_index}.jpg"
                cropped_image.save(output_dataset / class_name / output_name)

                copied_count[class_name] += 1


print("Cropped dataset organized successfully!")
print(f"Metal images: {copied_count['metal']}")
print(f"Paper images: {copied_count['paper']}")
print(f"Plastic images: {copied_count['plastic']}")
