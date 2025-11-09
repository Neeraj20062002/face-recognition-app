"""
test_model.py

Usage:
    python test_model.py <path_to_image>

Example:
    python test_model.py "dataset/modi/img1.jpg"

Outputs:
    - Prints predicted label and confidence
    - Appends a line to results/predictions.txt with timestamp, image, predicted_label, confidence
"""

import sys
from pathlib import Path
import json
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime

# Config
IMG_SIZE = (100, 100)
PCA_JOBLIB = Path("pca_model.joblib")
MODEL_PATH = Path("models") / "face_ann_model.h5"
LABELMAP_FILE = Path("label_map.json")
RESULTS_LOG = Path("results") / "predictions.txt"
RESULTS_LOG.parent.mkdir(exist_ok=True, parents=True)


def preprocess_image(img_path: Path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {img_path}")
    img = cv2.resize(img, IMG_SIZE)
    vec = img.flatten().astype(np.float32)
    return vec


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <path_to_image>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print("Image not found:", image_path)
        sys.exit(1)

    # Load PCA (scaler + pca)
    if not PCA_JOBLIB.exists():
        print("Missing PCA model file:", PCA_JOBLIB)
        sys.exit(1)
    pca_pack = joblib.load(str(PCA_JOBLIB))
    scaler = pca_pack.get("scaler")
    pca = pca_pack.get("pca")

    # Load label map
    if not LABELMAP_FILE.exists():
        print("Missing label_map.json")
        sys.exit(1)
    with open(LABELMAP_FILE, "r", encoding="utf-8") as f:
        label_map = json.load(f)  # keys are strings; later convert

    # Load ANN model
    if not MODEL_PATH.exists():
        print("Missing trained model:", MODEL_PATH)
        sys.exit(1)
    model = load_model(str(MODEL_PATH))

    # Preprocess image -> center -> pca transform -> predict
    vec = preprocess_image(image_path)
    vec_centered = scaler.transform([vec])    # shape (1, mn)
    vec_pca = pca.transform(vec_centered)     # shape (1, k)
    preds = model.predict(vec_pca)
    pred_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds, axis=1)[0])

    # label_map keys might be strings -> find mapping
    # label_map format created earlier should be { "0": "personname", ... } or similar
    # try both int and str keys
    label_str = None
    if str(pred_idx) in label_map:
        label_str = label_map[str(pred_idx)]
    elif pred_idx in label_map:
        label_str = label_map[pred_idx]
    else:
        # fallback: use smallest-distance index
        label_str = f"label_{pred_idx}"

    out_line = f"{datetime.now().isoformat()} | {image_path} | predicted: {label_str} | idx: {pred_idx} | conf: {confidence:.4f}"
    print(out_line)

    # Append to log
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(out_line + "\n")

    # Also save predicted label image copy to results/ (optional)
    try:
        res_img = cv2.imread(str(image_path))
        if res_img is not None:
            save_path = Path("results") / f"pred_{image_path.stem}_{pred_idx}.jpg"
            cv2.putText(res_img, f"{label_str} ({confidence:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imwrite(str(save_path), res_img)
    except Exception:
        pass


if __name__ == "__main__":
    main()
