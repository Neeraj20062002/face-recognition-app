# utils/augment_dataset.py
"""
Augment images in dataset/ into a new folder (default: dataset_aug/).
Does not overwrite original images.

Augmentations per image (deterministic + random):
 - horizontal flip
 - small rotation (-12, +12 degrees)
 - brightness change (-30 .. +30)
 - slight scaling/translation

Usage:
  python utils\augment_dataset.py --out_dir dataset_aug --copies 5

Output:
  dataset_aug/
    person1/
      orig_1.jpg
      aug_1_flip.jpg
      aug_1_rot+8.jpg
      ...
    person2/
      ...
Notes:
 - Requires: numpy, opencv-python
 - Keeps original aspect, resizes are not performed here (preprocessing will resize to 100x100).
"""
from pathlib import Path
import cv2
import numpy as np
import argparse
import random
from shutil import copy2

random.seed(42)
np.random.seed(42)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(folder: Path):
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in VALID_EXTS]

def augment_image(img):
    out = []
    # original (as-is)
    out.append(("orig", img.copy()))
    # horizontal flip
    out.append(("flip", cv2.flip(img, 1)))
    # small rotations
    for angle in (-12, -7, -3, 3, 7, 12):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rot = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        out.append((f"rot{angle}", rot))
    # brightness jitter
    for delta in (-30, -15, 15, 30):
        b = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        out.append((f"bright{delta}", b))
    # small scale + translate
    h, w = img.shape[:2]
    for s in (0.95, 1.05):
        M = cv2.getRotationMatrix2D((w/2, h/2), 0, s)
        scaled = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        out.append((f"scale{int(s*100)}", scaled))
    return out

def run_augment(input_root: Path, out_root: Path, copies: int):
    if not input_root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {input_root}")
    safe_mkdir(out_root)
    for person_folder in sorted([p for p in input_root.iterdir() if p.is_dir()]):
        person_out = out_root / person_folder.name
        safe_mkdir(person_out)
        imgs = list_images(person_folder)
        if not imgs:
            print(f"Warning: no images in {person_folder}, skipping")
            continue
        print(f"Processing {person_folder.name} ({len(imgs)} images)...")
        count = 0
        for img_path in imgs:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    print("  skip unreadable:", img_path.name)
                    continue
                base_stem = img_path.stem
                # Always copy original
                dst_orig = person_out / f"{base_stem}_orig{img_path.suffix}"
                copy2(img_path, dst_orig)
                count += 1
                # create unique augmented variants by sampling augmentations
                aug_pool = augment_image(img)
                # shuffle deterministically
                random.shuffle(aug_pool)
                # take 'copies-1' extra (because original already copied)
                needed = max(0, copies - 1)
                idx = 0
                for (tag, aug_img) in aug_pool:
                    if idx >= needed:
                        break
                    out_name = person_out / f"{base_stem}_aug{idx}_{tag}{img_path.suffix}"
                    cv2.imwrite(str(out_name), aug_img)
                    idx += 1
                    count += 1
            except Exception as e:
                print("  error processing", img_path, e)
        print(f"  created {count} files for {person_folder.name} -> {person_out}")
    print("Augmentation complete. Augmented data in:", out_root.resolve())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="dataset", help="Input dataset folder")
    parser.add_argument("--out_dir", default="dataset_aug", help="Output augmented dataset folder")
    parser.add_argument("--copies", type=int, default=5, help="Number of images per original (including original). e.g. 5 => 1 orig + 4 aug")
    args = parser.parse_args()
    run_augment(Path(args.in_dir), Path(args.out_dir), args.copies)
