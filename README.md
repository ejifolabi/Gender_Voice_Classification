# 🎤 Voice Gender Recognition

A professional and production-ready machine learning project that predicts a speaker's gender from their voice using audio files. It supports various audio formats and durations (minimum 3 seconds), and is robust against overfitting and underfitting. Deployed using Streamlit for easy web interaction.

---

## 📌 Features

✅ Accepts `.wav`, `.mp3`, `.flac`, `.ogg` audio formats  
✅ Works with both short (3s) and long audio  
✅ Uses MFCCs, Chroma, Spectral Contrast, ZCR, and RMS features  
✅ Offers two prediction modes: 
- **First 3 Seconds**
- **Full Audio Analysis**

✅ Visualizes waveform and spectrogram  
✅ Trained with Random Forest on gender-labeled voice dataset  
✅ Accurate, clean, and efficient (99%+ test accuracy).

---

## 🏗️ Project Structure

```bash
voice-gender-recognition/
├── app.py                  # Streamlit deployment code
├── models/
│   └── gender_model.pkl    # Trained ML model (RandomForest)
├── requirements.txt        # Project dependencies
├── README.md               # Project overview and usage

---
