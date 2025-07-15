# 🎙️ Voice Gender Classification App

A professional-grade machine learning web application that predicts the **gender of a speaker** from an audio sample (≥ 3 seconds). Built with **Python, Streamlit, librosa**, and **scikit-learn**, it accepts multiple audio formats (`.wav`, `.mp3`, `.ogg`, `.flac`, `.aac`) and is deployed live on the web.

🌐 **Live App**: [Try it Now](https://gendervoiceclassification-26dw4k2cfugsakhys6vyd8.streamlit.app/)

---

## 📌 Features

- ✅ Accepts multiple audio formats: `.wav`, `.mp3`, `.ogg`, `.aac`, `.flac`
- ✅ Handles both short and long audio
- ✅ Predicts using:
  - First 3 seconds
  - OR full audio with sliding window voting
- ✅ Shows audio waveform and spectrogram
- ✅ Deployed live with Streamlit Cloud
- ✅ Clean and intuitive UI
- ✅ Real-time gender prediction (Male 👨 / Female 👩)

---

## 📂 Project Structure

voice-gender-recognition/

├── app.py # Streamlit application

├── models/

│ └── gender_model.pkl # Trained RandomForest model

├── requirements.txt # Python dependencies

├── packages.txt # System dependencies (for FFmpeg)

└── README.md # Project documentation

---

## 🎯 How It Works

1. User uploads an audio file (≥ 3 seconds)
2. The app converts it to `.wav` internally using **pydub**
3. Extracts features:
   - MFCCs (13)
   - Chroma STFT
   - Spectral Contrast
   - Zero Crossing Rate
   - RMS Energy
4. Predicts gender using a pre-trained **Random Forest Classifier**
5. Visualizes waveform + spectrogram
6. Displays prediction with emoji

---

## 📊 Dataset

- **Name**: [Gender Recognition by Voice](https://www.kaggle.com/datasets/murtadhanajim/gender-recognition-by-voiceoriginal)
- **Source**: Kaggle
- **Format**: `.wav` files in `male/` and `female/` folders
- **Sampling Rate**: 16,000 Hz
- **Class Labels**: `0` for male, `1` for female

---

## 🧠 Model Training

- **Algorithm**: RandomForestClassifier (`scikit-learn`)
- **Input Features**: 13 MFCC + Chroma + Contrast + ZCR + RMS
- **Split**: 80% training, 20% test
- **Accuracy**: `~99%` on unseen test data
- **Preprocessing**:
  - Converted all audio to mono, 16kHz
  - Fixed/truncated length to 3 seconds
  - Standardized feature dimensions

---

## 🚀 Deployment

### Streamlit Cloud Setup:

1. ✅ `requirements.txt` for Python packages
2. ✅ `packages.txt` to install FFmpeg on the server
3. ✅ Model stored in `models/gender_model.pkl`
4. ✅ Deployed from GitHub

App runs fully in-browser — **no installation needed by end-users**.

---

## 🛠️ Local Setup

```bash
# Clone repo
git clone https://github.com/your-username/voice-gender-recognition.git
cd voice-gender-recognition

# Install Python dependencies
pip install -r requirements.txt

# Optional (for local audio format support)
# Windows: https://www.gyan.dev/ffmpeg/builds/
# Mac: brew install ffmpeg
# Linux: sudo apt install ffmpeg

# Run app
streamlit run app.py
