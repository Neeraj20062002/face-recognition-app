# 🤖 Face Recognition App (PCA + SVM + Streamlit)

### 🧠 Overview

A face recognition system using **Principal Component Analysis (PCA)** for dimensionality reduction and **Support Vector Machine (SVM)** for classification.  
Includes an interactive **Streamlit GUI** for uploads, webcam, and live predictions.

---

## ⚙️ Features

- PCA-based Eigenfaces generation
- SVM classification with adjustable confidence threshold
- Real-time prediction (upload / webcam)
- Accuracy charts and dataset visualization
- Prediction logging and reports
- Modular, well-structured Python code

---

## 📂 Project Structure

face_recognition_app/
│
├── dataset/ # Training images
├── models/ # Saved models (.joblib, .h5)
├── results/ # PCA outputs, charts, logs
├── utils/ # Preprocessing, PCA, augmentation helpers
│ ├── preprocessing.py
│ ├── pca_module.py
│ ├── evaluation.py
│ └── augment_dataset.py
│
├── demo_app.py # Streamlit GUI
├── train_svm.py # PCA + SVM trainer
├── svm_predict.py # CLI predictor
├── requirements.txt # Dependencies
└── README.md # Documentation

yaml
Copy code

---

## 🚀 Run Locally

```bash
1️⃣ Clone the repo
git clone https://github.com/Neeraj20062002/face-recognition-app.git
cd face-recognition-app
2️⃣ Install requirements
bash
Copy code
pip install -r requirements.txt
3️⃣ Start Streamlit app
bash
Copy code
streamlit run demo_app.py
📊 Model Summary
Metric	Value
Accuracy	97.5 % (SVM)
PCA Components	10
Classes	modi · prabhas · robert_dowyne
Dataset	Augmented (100 images)

🖼 Example
bash
Copy code
python svm_predict.py "dataset/modi/1_orig.jpg"
→ Predicted: modi | Confidence: 93.3 %

Streamlit Dashboard :

Upload or capture image
View prediction + confidence
Interactive accuracy & class plots

🧾 Reports :

results/svm_report.txt → Evaluation
results/training_history.png → Accuracy curve
results/predictions_log.csv → Logs

```
