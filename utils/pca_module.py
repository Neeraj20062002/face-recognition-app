"""
utils/pca_module.py

Usage:
    - Ensure faces.npy, labels.npy, label_map.json exist in project root (created by preprocessing.py).
    - Run: python utils\pca_module.py
Outputs (saved to project root / results/):
    - pca_model.joblib          -> saved PCA object
    - X_train_pca.npy, X_test_pca.npy
    - y_train.npy, y_test.npy
    - eigenface_0.png, eigenface_1.png, ... (in results/)
    - summary printed to console
"""

from pathlib import Path
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

# Config
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
FACES_FILE = Path("faces.npy")
LABELS_FILE = Path("labels.npy")
LABELMAP_FILE = Path("label_map.json")

TEST_SIZE = 0.40
RANDOM_STATE = 42
DEFAULT_K = 10  # desired number of principal components; will be capped by data shape
EIGENFACE_SAVE_COUNT = 8  # how many eigenfaces to save as images


def load_data():
    if not FACES_FILE.exists() or not LABELS_FILE.exists():
        raise FileNotFoundError("faces.npy or labels.npy not found. Run preprocessing first.")
    faces = np.load(FACES_FILE)       # shape (N, mn)
    labels = np.load(LABELS_FILE)     # shape (N,)
    with open(LABELMAP_FILE, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    return faces, labels, label_map


def compute_pca(faces: np.ndarray, k: int):
    """
    faces: shape (N, mn)
    returns: scaler, pca (fitted), X_pca (transformed full data)
    """
    # Standardize (zero-mean) per feature (pixel)
    scaler = StandardScaler(with_mean=True, with_std=False)  # only mean centering
    faces_centered = scaler.fit_transform(faces)  # shape (N, mn)

    # Decide k safely
    n_samples, n_features = faces_centered.shape
    max_k = min(n_samples - 1, n_features)
    if k > max_k:
        print(f"Warning: requested k={k} is too large for data; using k={max_k} instead.")
        k = max_k

    pca = PCA(n_components=k, svd_solver="auto", whiten=False, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(faces_centered)  # shape (N, k)

    return scaler, pca, X_pca


def save_eigenfaces(pca: PCA, scaler_mean: np.ndarray, img_shape=(100, 100), count=EIGENFACE_SAVE_COUNT):
    """
    Save top 'count' eigenfaces as images in results/.
    pca.components_ shape: (k, mn)
    scaler_mean: mean vector (len mn) from StandardScaler
    """
    comps = pca.components_  # (k, mn)
    for i in range(min(count, comps.shape[0])):
        ef = comps[i].reshape(img_shape)  # eigenface (may contain negative values)
        # Normalize to 0-255 for saving
        ef_norm = ef - ef.min()
        if ef_norm.max() != 0:
            ef_norm = ef_norm / ef_norm.max()
        ef_img = (ef_norm * 255).astype(np.uint8)
        out_path = RESULTS_DIR / f"eigenface_{i}.png"
        plt.imsave(out_path, ef_img, cmap="gray")
    print(f"Saved {min(count, comps.shape[0])} eigenfaces to {RESULTS_DIR.resolve()}")


def main():
    print("Loading preprocessed data...")
    faces, labels, label_map = load_data()
    N, vec_size = faces.shape
    print(f"Data shape: {faces.shape} (N={N}, vector_size={vec_size})")
    # default safe k
    safe_k = DEFAULT_K
    if safe_k > (N - 1):
        safe_k = max(1, N - 1)
        print(f"Adjusted DEFAULT_K to {safe_k} due to small dataset (N).")

    print(f"Computing PCA with k={safe_k} ...")
    scaler, pca, X_pca = compute_pca(faces, safe_k)

    # Split into train/test on transformed data
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(
        X_pca, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels if len(np.unique(labels))>1 else None
    )

    # Save artifacts
    joblib.dump({"scaler": scaler, "pca": pca}, "pca_model.joblib")
    np.save("X_train_pca.npy", X_train_pca)
    np.save("X_test_pca.npy", X_test_pca)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    with open("label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print("Saved PCA model (pca_model.joblib) and train/test PCA arrays.")
    # Save eigenfaces (visual)
    # Need to know original image shape; we used 100x100 in preprocessing by default
    img_side = int(np.sqrt(vec_size))
    if img_side * img_side != vec_size:
        print("Warning: vector size is not a perfect square; eigenface images may not visualize correctly.")
        img_shape = (100, 100)
    else:
        img_shape = (img_side, img_side)

    save_eigenfaces(pca, scaler.mean_, img_shape=img_shape, count=EIGENFACE_SAVE_COUNT)

    # Summary
    print("Summary:")
    print(f"  PCA components kept: {pca.n_components_}")
    print(f"  Explained variance ratio (first 10): {np.round(pca.explained_variance_ratio_[:10], 4)}")
    print(f"  X_train_pca shape: {X_train_pca.shape}")
    print(f"  X_test_pca shape: {X_test_pca.shape}")
    print("PCA step complete.")


if __name__ == "__main__":
    main()
