# train_svm.py
"""
Train an SVM on PCA features and save the model.

Outputs:
 - svm_model.joblib     (saved SVM)
 - results/svm_report.txt
"""

from pathlib import Path
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
SVM_OUT = Path("svm_model.joblib")
REPORT = RESULTS / "svm_report.txt"

# files created by pca_module.py
X_TRAIN = Path("X_train_pca.npy")
X_TEST = Path("X_test_pca.npy")
Y_TRAIN = Path("y_train.npy")
Y_TEST = Path("y_test.npy")
LABELMAP = Path("label_map.json")

def load_data():
    X_train = np.load(X_TRAIN)
    X_test = np.load(X_TEST)
    y_train = np.load(Y_TRAIN)
    y_test = np.load(Y_TEST)
    with open(LABELMAP, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    return X_train, X_test, y_train, y_test, label_map

def main():
    X_train, X_test, y_train, y_test, label_map = load_data()
    print(f"Shapes: X_train {X_train.shape}, X_test {X_test.shape}")
    clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")
    # Save model
    joblib.dump({"svm": clf, "label_map": label_map}, SVM_OUT)
    # Report
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(f"Test accuracy: {acc:.6f}\n\n")
        f.write("Classification report:\n")
        f.write(classification_report(y_test, preds))
        f.write("\nConfusion matrix:\n")
        f.write(str(confusion_matrix(y_test, preds)))
        f.write("\nLabel map:\n")
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"Saved SVM model to {SVM_OUT} and report to {REPORT}")

if __name__ == "__main__":
    main()
