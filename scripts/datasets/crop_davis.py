#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Dataset paths
BASE_DIR = Path("data", "davis_2017", "DAVIS")
IMAGE_DIR = Path(BASE_DIR, "JPEGImages", "Full-Resolution")
ANNOTATION_DIR = Path(BASE_DIR, "Annotations", "Full-Resolution")
CROPPED_DIR = Path(BASE_DIR, "CroppedImages")
LABEL_PATH = Path(BASE_DIR, "davis_semantics.json")

# Color to index mapping based on specified RGB values
COLOR_TO_INDEX = {
    (128, 0, 0): "1",
    (0, 128, 0): "2",
    (128, 128, 0): "3",
    (0, 0, 128): "4",
    (128, 0, 128): "5",
    (0, 128, 128): "6",
    (128, 128, 128): "7",
    (64, 0, 0): "8",
    (0, 0, 0): "background",
}


# Function to get bounding box from mask
def get_bounding_box(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return x_min, y_min, x_max, y_max


def main():
    # Create the output directory
    CROPPED_DIR.mkdir(exist_ok=True)

    # Load object category labels
    with open(LABEL_PATH) as label_file:
        label_dict = json.load(label_file)

    # Process each sequence
    for sequence_image_dir in tqdm(list(IMAGE_DIR.iterdir()), ncols=0):
        sequence_name = sequence_image_dir.name
        sequence_cropped_dir = CROPPED_DIR / sequence_name
        sequence_cropped_dir.mkdir(exist_ok=True)
        object_id_tracker = {}

        for image_path in sequence_image_dir.iterdir():
            annotation_path = ANNOTATION_DIR / sequence_name / f"{image_path.stem}.png"
            image = Image.open(image_path)
            annotations = np.array(Image.open(annotation_path).convert("RGB"))
            for color, index in COLOR_TO_INDEX.items():
                if index == "background":
                    continue

                # Maintain the object ID across frames
                if color not in object_id_tracker:
                    object_id_tracker[color] = len(object_id_tracker) + 1
                object_id = object_id_tracker[color]

                mask = np.all(annotations == np.array(color), axis=-1)
                if mask.any():
                    cropped_image = image.crop(get_bounding_box(mask))
                    label = label_dict[sequence_name][index]
                    cropped_path = (
                        f"{image_path.stem}_{sequence_name}_{label}_{object_id}.png"
                    )
                    cropped_image.save(sequence_cropped_dir / cropped_path)


if __name__ == "__main__":
    main()
