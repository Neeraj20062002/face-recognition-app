# svm_predict_prob.py
import sys
from pathlib import Path
import joblib, json, numpy as np, cv2
from datetime import datetime

PCA_JOBLIB = Path("pca_model.joblib")
SVM_JOBLIB = Path("svm_model.joblib")
LABELMAP = Path("label_map.json")
IMG_SIZE=(100,100)
PROB_THRESHOLD = 0.60   # choose e.g. 0.6; adjust

def preprocess(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, IMG_SIZE)
    return img.flatten().astype("float32")

if len(sys.argv)<2:
    print("Usage: python svm_predict_prob.py path/to/image.jpg")
    raise SystemExit(1)

imgp = Path(sys.argv[1])
pca_pack = joblib.load(PCA_JOBLIB)
scaler = pca_pack["scaler"]; pca = pca_pack["pca"]
svm_pack = joblib.load(SVM_JOBLIB)
clf = svm_pack["svm"]
with open(LABELMAP,"r",encoding="utf-8") as f: label_map = json.load(f)

vec = preprocess(imgp)
vecc = scaler.transform([vec])
vecp = pca.transform(vecc)
probs = clf.predict_proba(vecp)[0]
pred_idx = int(np.argmax(probs))
pred_prob = float(np.max(probs))
label = label_map.get(str(pred_idx), label_map.get(pred_idx, f"label_{pred_idx}"))
if pred_prob < PROB_THRESHOLD:
    label = "Unknown"

nearest_dist = float(np.min(np.linalg.norm(np.load("X_train_pca.npy") - vecp, axis=1)))
print(f"{datetime.now().isoformat()} | {imgp} | predicted: {label} | idx: {pred_idx} | prob: {pred_prob:.3f} | nearest_dist: {nearest_dist:.3f}")
