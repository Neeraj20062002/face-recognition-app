import streamlit as st
import numpy as np
import joblib
import cv2
import os
from datetime import datetime
from utils.preprocessing import preprocess_image_for_pca
from sklearn.decomposition import PCA

# ============================
# CONFIG
# ============================
st.set_page_config(page_title="Face Recognition Demo", page_icon="🤖", layout="wide")
MODEL_PATH = "svm_model.joblib"
PCA_PATH = "pca_model.joblib"
LABEL_MAP_PATH = "label_map.json"
PROB_THRESHOLD = 0.45  # lower = less strict; higher = more cautious

# ============================
# LOAD MODELS
# ============================
@st.cache_resource
def load_models():
    svm = joblib.load(MODEL_PATH)
    pca = joblib.load(PCA_PATH)
    import json
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}
    return svm, pca, label_map

svm, pca, label_map = load_models()

# ============================
# HELPERS
# ============================
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 100)).flatten()
    return resized

def predict_image(img_array):
    vec = preprocess_image(img_array).reshape(1, -1)
    vec_pca = pca.transform(vec)
    probas = svm.predict_proba(vec_pca)[0]
    idx = np.argmax(probas)
    prob = probas[idx]
    name = label_map[idx] if prob >= PROB_THRESHOLD else "Unknown"
    return name, prob, idx

# ============================
# UI
# ============================
st.title("🤖 Face Recognition using PCA + SVM")
st.markdown("Upload a face image or use the webcam to predict the identity.")

tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Webcam Capture"])

# UPLOAD TAB
with tab1:
    uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
        if st.button("🔍 Predict from Uploaded Image"):
            name, prob, idx = predict_image(img)
            st.success(f"**Prediction:** {name}")
            st.metric(label="Confidence", value=f"{prob*100:.2f}%")
            st.caption(f"Predicted label index: {idx}")

# WEBCAM TAB
with tab2:
    pic = st.camera_input("Take a picture")
    if pic is not None:
        file_bytes = np.asarray(bytearray(pic.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Captured Image", use_container_width=True)
        if st.button("🔍 Predict from Camera Capture"):
            name, prob, idx = predict_image(img)
            st.success(f"**Prediction:** {name}")
            st.metric(label="Confidence", value=f"{prob*100:.2f}%")
            st.caption(f"Predicted label index: {idx}")

st.divider()
st.caption("💡 Tip: Adjust PROB_THRESHOLD in code to make predictions stricter or more flexible.")
st.caption("Developed by Neeraj · ISTUDIO Internship · 2025")
