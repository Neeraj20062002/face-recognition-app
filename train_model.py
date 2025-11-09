"""
train_model.py

Usage:
    python train_model.py

Outputs:
    - models/face_ann_model.h5        (saved Keras model)
    - results/training_history.png    (accuracy & loss plots)
    - results/train_report.txt        (summary: train/val acc/loss)
"""

from pathlib import Path
import numpy as np
import json
import os

# --- ML libs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

# Paths
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Files produced by pca_module.py
X_TRAIN_FILE = Path("X_train_pca.npy")
X_TEST_FILE = Path("X_test_pca.npy")
Y_TRAIN_FILE = Path("y_train.npy")
Y_TEST_FILE = Path("y_test.npy")
LABELMAP_FILE = Path("label_map.json")

# Output names
MODEL_PATH = MODELS_DIR / "face_ann_model.h5"
HISTORY_PLOT = RESULTS_DIR / "training_history.png"
REPORT_TXT = RESULTS_DIR / "train_report.txt"

# Hyperparams (tweakable)
EPOCHS = 60
BATCH_SIZE = 4
VALIDATION_SPLIT = 0.2   # will be used on training set for internal val
RANDOM_SEED = 42
LR = 1e-3
DENSE_UNITS = [128, 64]  # hidden layers


def load_data():
    X_train = np.load(X_TRAIN_FILE)
    X_test = np.load(X_TEST_FILE)
    y_train = np.load(Y_TRAIN_FILE)
    y_test = np.load(Y_TEST_FILE)
    with open(LABELMAP_FILE, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    return X_train, X_test, y_train, y_test, label_map


def build_model(input_dim: int, num_classes: int):
    tf.random.set_seed(RANDOM_SEED)
    model = Sequential()
    model.add(Dense(DENSE_UNITS[0], activation="relu", input_shape=(input_dim,)))
    model.add(Dropout(0.2))
    model.add(Dense(DENSE_UNITS[1], activation="relu"))
    model.add(Dropout(0.15))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def plot_history(history, out_path: Path):
    plt.figure(figsize=(10, 4))
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    print("Loading PCA data...")
    X_train, X_test, y_train, y_test, label_map = load_data()
    num_classes = len(np.unique(y_train))
    print(f"Shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Num classes: {num_classes}, label_map: {label_map}")

    # One-hot
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    model = build_model(input_dim=X_train.shape[1], num_classes=num_classes)
    model.summary()

    # Callbacks
    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)
    ckpt = ModelCheckpoint(str(MODEL_PATH), monitor="val_loss", save_best_only=True, verbose=1)

    # Train
    history = model.fit(
        X_train, y_train_cat,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es, ckpt],
        verbose=2
    )

    # Evaluate on X_test
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")

    # Save final model (ModelCheckpoint already saved best)
    if not MODEL_PATH.exists():
        model.save(MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH.resolve()}")

    # Plots & report
    plot_history(history, HISTORY_PLOT)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(f"Test loss: {loss:.6f}\nTest accuracy: {acc:.6f}\n")
        f.write(f"Train shape: {X_train.shape}\nTest shape: {X_test.shape}\n")
        f.write("Label map:\n")
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"Saved training history plot to {HISTORY_PLOT} and report to {REPORT_TXT}")

    print("Training complete.")


if __name__ == "__main__":
    main()
