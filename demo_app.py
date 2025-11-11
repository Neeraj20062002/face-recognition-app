import streamlit as st
import numpy as np
import joblib
import cv2
import os
import pandas as pd
import plotly.express as px
from datetime import datetime

# ============================
# CONFIG
# ============================
st.set_page_config(page_title="Face Recognition Demo", page_icon="🤖", layout="wide")
MODEL_PATH = "svm_model.joblib"
PCA_PATH = "pca_model.joblib"
LABEL_MAP_PATH = "label_map.json"
PROB_THRESHOLD = 0.45  # lower = less strict; higher = more cautious
IMG_SIZE = (100, 100)

# ============================
# LOAD MODELS (robust)
# ============================
@st.cache_resource
def load_models():
    # Load SVM
    svm_pack = joblib.load(MODEL_PATH)
    if isinstance(svm_pack, dict):
        svm = svm_pack.get("svm", svm_pack.get("model", None))
    else:
        svm = svm_pack

    # Load PCA pack: could contain both scaler + pca
    pca_pack = joblib.load(PCA_PATH)
    scaler = None
    pca = None
    if isinstance(pca_pack, dict):
        scaler = pca_pack.get("scaler", pca_pack.get("scaler_", None))
        pca = pca_pack.get("pca", pca_pack.get("model", None))
    else:
        pca = pca_pack

    # Load Label map
    import json
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}

    return svm, scaler, pca, label_map


svm, scaler, pca, label_map = load_models()

if svm is None:
    raise RuntimeError("SVM model not found or invalid.")
if pca is None:
    raise RuntimeError("PCA model not found or invalid.")

# ============================
# HELPERS
# ============================
def preprocess_image(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE).flatten().astype(np.float32)
    return resized

def apply_scaler(vec_2d):
    if scaler is None:
        return vec_2d
    try:
        return scaler.transform(vec_2d)
    except Exception:
        if hasattr(scaler, "mean_"):
            return vec_2d - scaler.mean_
        try:
            mean_vec = np.asarray(scaler)
            if mean_vec.shape == (vec_2d.shape[1],):
                return vec_2d - mean_vec.reshape(1, -1)
        except Exception:
            pass
    return vec_2d

def predict_image(img_bgr):
    vec = preprocess_image(img_bgr).reshape(1, -1)
    vec_centered = apply_scaler(vec)
    vec_pca = pca.transform(vec_centered)
    if not hasattr(svm, "predict_proba"):
        raise RuntimeError("SVM model doesn't support predict_proba (retrain with probability=True).")

    probs = svm.predict_proba(vec_pca)[0]
    idx = int(np.argmax(probs))
    prob = float(np.max(probs))
    name = label_map.get(idx, f"label_{idx}")
    display_name = name if prob >= PROB_THRESHOLD else "Unknown"
    return display_name, prob, idx

# ============================
# UI
# ============================
st.title("🤖 Face Recognition using PCA + SVM")
st.markdown("Upload a face image or use the webcam to predict the identity.")

tab1, tab2, tab3 = st.tabs(["📁 Upload Image", "📷 Webcam Capture", "📊 Model Insights"])

# UPLOAD TAB
with tab1:
    uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Unable to decode image. Try a different file.")
        else:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
            if st.button("🔍 Predict from Uploaded Image"):
                try:
                    name, prob, idx = predict_image(img)
                    st.success(f"**Prediction:** {name}")
                    st.metric(label="Confidence", value=f"{prob*100:.2f}%")
                    st.caption(f"Predicted label index: {idx}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

# WEBCAM TAB
with tab2:
    pic = st.camera_input("Take a picture")
    if pic is not None:
        file_bytes = np.asarray(bytearray(pic.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Unable to decode camera image.")
        else:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Captured Image", use_container_width=True)
            if st.button("🔍 Predict from Camera Capture"):
                try:
                    name, prob, idx = predict_image(img)
                    st.success(f"**Prediction:** {name}")
                    st.metric(label="Confidence", value=f"{prob*100:.2f}%")
                    st.caption(f"Predicted label index: {idx}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

# ============================
# MODEL INSIGHTS TAB
# ============================
with tab3:
    st.subheader("📈 Model Performance Overview")

    report_path = os.path.join("results", "svm_report.txt")
    if os.path.exists(report_path):
        st.success("Loaded latest SVM training report ✅")
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()
        st.text_area("Training Report", report_text, height=250)

        # Extract accuracy
        lines = [line.strip() for line in report_text.split("\n") if line.strip()]
        acc_line = [l for l in lines if "accuracy" in l.lower()]
        if acc_line:
            st.metric(label="Overall Test Accuracy", value=acc_line[0].split()[-1])
    else:
        st.warning("No SVM report found yet. Train the model first.")

    st.divider()
    st.subheader("🔍 Class Distribution")

    class_counts = {}
    for class_name in os.listdir("dataset"):
        class_dir = os.path.join("dataset", class_name)
        if os.path.isdir(class_dir):
            n = len([f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[class_name] = n

    df_counts = pd.DataFrame({
        "Class": list(class_counts.keys()),
        "Images": list(class_counts.values())
    })

    fig_bar = px.bar(df_counts, x="Class", y="Images", text="Images",
                     color="Class", title="Number of Training Images per Class")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    st.subheader("🧠 Accuracy by Class (from Confusion Matrix)")

    conf_csv = os.path.join("results", "confusion_matrix.csv")
    if os.path.exists(conf_csv):
        cm = pd.read_csv(conf_csv, index_col=0)
        st.dataframe(cm)
        fig_cm = px.imshow(cm, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                           title="Confusion Matrix Heatmap")
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("Confusion matrix CSV not found yet (optional file).")

# ============================
# ℹ️ Model Information Footer
# ============================
st.divider()
st.subheader("📊 Model Information")

with st.expander("Show Model Details"):
    st.write("**Model Type:** PCA + SVM")
    st.write("**Accuracy:** 97.5 % (evaluated on 100 images)")
    st.write("**PCA Components Used:** 10")
    st.write("**Classes:** modi · prabhas · robert_dowyne")
    st.write("**Dataset:** Augmented facial dataset (100 images total)")
    st.caption("Last trained on: 2025-11-09")
