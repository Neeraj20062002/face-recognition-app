# utils/quick_k_eval.py
"""
Quick evaluator: finds accuracy vs PCA k using an SVM classifier.
Usage:
    python utils/quick_k_eval.py

Outputs:
    - results/accuracy_vs_k_svm.png
    - prints table of k -> test accuracy
Notes:
    - Uses same preprocessing (100x100 flattened vectors).
    - Good to run before retraining ANN to pick a reasonable k.
"""
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

ROOT = Path(".")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

FACES_FILE = ROOT / "faces.npy"
LABELS_FILE = ROOT / "labels.npy"
TEST_SIZE = 0.40
RANDOM_STATE = 42

K_LIST = [3, 5, 8, 10, 12, 15]  # adjust if you want more values

def load_data():
    faces = np.load(FACES_FILE)
    labels = np.load(LABELS_FILE)
    return faces, labels

def eval_for_k(faces, labels, k):
    # mean-center (same as PCA module)
    scaler = StandardScaler(with_mean=True, with_std=False)
    faces_centered = scaler.fit_transform(faces)

    # PCA
    max_k = min(faces_centered.shape[0] - 1, faces_centered.shape[1])
    k_use = min(k, max_k)
    if k != k_use:
        print(f"Adjusted k={k} -> k_use={k_use} due to data size.")
    pca = PCA(n_components=k_use, svd_solver="auto", random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(faces_centered)

    # Split and SVM
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels if len(np.unique(labels))>1 else None
    )

    clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=False, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc

def main():
    faces, labels = load_data()
    results = []
    print("Evaluating accuracy vs k (SVM) ...")
    for k in K_LIST:
        try:
            acc = eval_for_k(faces, labels, k)
            results.append((k, acc))
            print(f"  k={k:2d} -> test acc = {acc:.3f}")
        except Exception as e:
            print(f"  k={k:2d} -> error: {e}")

    # Plot
    ks = [r[0] for r in results]
    accs = [r[1] for r in results]
    plt.figure(figsize=(6,4))
    plt.plot(ks, accs, marker='o')
    plt.title("Accuracy vs PCA k (SVM classifier)")
    plt.xlabel("k (principal components)")
    plt.ylabel("Test accuracy")
    plt.grid(True)
    out_png = RESULTS / "accuracy_vs_k_svm.png"
    plt.savefig(out_png)
    plt.close()
    print(f"Saved plot to {out_png}")

if __name__ == "__main__":
    main()
