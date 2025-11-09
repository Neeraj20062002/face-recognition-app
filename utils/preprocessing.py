"""
utils/preprocessing.py

Usage:
    - Put your dataset organized as:
        dataset/
          person1/
            img1.jpg
            img2.png
          person2/
            ...
    - Run: python utils/preprocessing.py
    - Output files produced in project root:
        faces.npy       -> numpy array shape (N, mn)
        labels.npy      -> numpy array shape (N,)
        label_map.json  -> mapping {label_id: person_folder_name}
"""

import os
import json
from pathlib import Path
from typing import Tuple, List, Dict

import cv2
import numpy as np

# Config
DATASET_DIR = Path("dataset")
IMG_SIZE = (100, 100)   # (width, height) — you can change this consistently across project
OUTPUT_FACES = Path("faces.npy")
OUTPUT_LABELS = Path("labels.npy")
OUTPUT_LABELMAP = Path("label_map.json")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_images_from_dataset(dataset_dir: Path = DATASET_DIR,
                             img_size: Tuple[int, int] = IMG_SIZE
                             ) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Loads images from dataset_dir.
    Returns:
        faces: np.ndarray shape (N, mn)  -- flattened grayscale images
        labels: np.ndarray shape (N,)
        label_map: dict {label_id: folder_name}
    """
    faces: List[np.ndarray] = []
    labels: List[int] = []
    label_map: Dict[int, str] = {}

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir.resolve()}")

    label_id = 0
    # Sort folder names to make label assignment deterministic
    for person_folder in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        person_name = person_folder.name
        label_map[label_id] = person_name
        files = sorted([f for f in person_folder.iterdir() if f.suffix.lower() in VALID_EXTS])
        if not files:
            print(f"Warning: no valid image files found in {person_folder}")
        for img_path in files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: unable to read {img_path}, skipping.")
                continue
            # Resize to fixed size
            img_resized = cv2.resize(img, img_size)
            # Flatten (row-major) to vector
            faces.append(img_resized.flatten().astype(np.float32))
            labels.append(label_id)
        label_id += 1

    if len(faces) == 0:
        raise RuntimeError("No images loaded. Check dataset directory and supported file extensions.")

    faces_arr = np.vstack([f[np.newaxis, :] for f in faces])   # shape (N, mn)
    labels_arr = np.array(labels, dtype=np.int32)

    return faces_arr, labels_arr, label_map


def save_outputs(faces: np.ndarray, labels: np.ndarray, label_map: Dict[int, str]):
    np.save(OUTPUT_FACES, faces)
    np.save(OUTPUT_LABELS, labels)
    with open(OUTPUT_LABELMAP, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"Saved: {OUTPUT_FACES} ({faces.shape}), {OUTPUT_LABELS} ({labels.shape}), {OUTPUT_LABELMAP}")


def main():
    print("Loading images from dataset...")
    faces, labels, label_map = load_images_from_dataset()
    print(f"Loaded {faces.shape[0]} images. Image vector size: {faces.shape[1]}")
    # Basic sanity print
    counts = {}
    for v in labels:
        counts[v] = counts.get(v, 0) + 1
    print("Per-class image counts:")
    for k, v in counts.items():
        print(f"  {k}: {v} images  -> {label_map[k]}")

    # Save
    save_outputs(faces, labels, label_map)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
