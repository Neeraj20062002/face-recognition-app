# eval_all_images.py
from pathlib import Path
import joblib, json, numpy as np, cv2
pca_pack = joblib.load("pca_model.joblib")
scaler = pca_pack["scaler"]; pca = pca_pack["pca"]
svm_pack = joblib.load("svm_model.joblib")
clf = svm_pack["svm"]
with open("label_map.json","r",encoding="utf-8") as f: label_map = json.load(f)
root = Path("dataset")
for person in sorted([d for d in root.iterdir() if d.is_dir()]):
    for img in sorted([f for f in person.iterdir() if f.is_file()]):
        img_gray = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        if img_gray is None: continue
        img_r = cv2.resize(img_gray,(100,100)).flatten().astype("float32")
        v = scaler.transform([img_r])
        vp = pca.transform(v)
        pred = int(clf.predict(vp)[0])
        dist = float(np.min(np.linalg.norm(np.load("X_train_pca.npy")-vp,axis=1)))
        print(f"{person.name}/{img.name} -> predicted: {label_map.get(str(pred),pred)} | nearest_dist: {dist:.3f}")
