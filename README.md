# 🎙️ Voice Gender Classification Using AI and Signal Processing

This project uses **real human voice audio recordings** to classify gender (Male/Female) using **feature extraction techniques** and a **Random Forest machine learning model**. Built with Python, it demonstrates the practical application of signal processing in intelligent voice systems.

## 📦 Dataset

- **Name**: [Gender Recognition by Voice (Original)](https://www.kaggle.com/datasets/murtadhanajim/gender-recognition-by-voiceoriginal)
- **Format**: `.wav` audio files

## 🧠 Model Overview

- **Algorithm**: Random Forest Classifier
- **Features Extracted**:
- MFCC (Mel Frequency Cepstral Coefficients)
- Chroma Frequencies
- Spectral Centroid
- Zero Crossing Rate

## 🧪 Model Performance

| Metric        | Female      | Male        |
|---------------|-------------|-------------|
| Precision     | **0.99**    | **0.99**    |
| Recall        | **0.98**    | **0.99**    |
| F1-Score      | **0.98**    | **0.99**    |
| Overall Accuracy | **99%** |

✅ Confusion matrix and classification report are printed during training.

## 🚀 How It Works

1. Download `.wav` dataset from Kaggle using `kagglehub`
2. Extract features from each audio file using `librosa`
3. Train a `RandomForestClassifier` from `scikit-learn`
4. Save the model to disk
5. Predict gender from new `.wav` voice samples

---

## 🧰 Tech Stack

| Tool          | Use                                 |
|---------------|--------------------------------------|
| `Python 3.8+` | Programming Language                |
| `librosa`     | Audio feature extraction            |
| `scikit-learn`| Machine Learning                    |
| `kagglehub`   | Downloading dataset from Kaggle     |
| `joblib`      | Model serialization                 |

## ✍️ Author
Emmanuel Oludare Ejifolabi

AI & Signal Processing Enthusiast

[LinkedIn](https://www.linkedin.com/in/emmagee001) | [GitHub](https://www.github.com/ejifolabi)


### **NOTE:** Work is ongoing to improve the predictions because the model is slightly overfitted.
