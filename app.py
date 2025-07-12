# app.py

import streamlit as st
import numpy as np
import librosa
import joblib
import io
from pydub import AudioSegment
import os

# Point to ffmpeg path (for Streamlit Cloud compatibility)
AudioSegment.converter = "/usr/bin/ffmpeg"

# === Load Trained Model ===
@st.cache_resource
def load_model():
    model_path = "models/voice_gender_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Model not found.")
        return None
    return joblib.load(model_path)

model = load_model()

# === Feature Extraction ===
def extract_features(uploaded_file):
    try:
        st.write("📎 File name:", uploaded_file.name)
        st.write("📄 MIME type:", uploaded_file.type)

        # Load file into memory using pydub
        audio = AudioSegment.from_file(uploaded_file)

        # Export to .wav without forcing mono
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # Load using librosa
        y, sr = librosa.load(wav_io, sr=22050, mono=True)  # mono=True for safe feature extraction

        duration = librosa.get_duration(y=y, sr=sr)
        st.write(f"📏 Duration: {duration:.2f} sec")

        if duration < 0.5:
            st.warning("⚠️ Audio too short (< 0.5s). Please upload a longer clip.")
            return None

        if y.size == 0:
            st.warning("⚠️ No valid audio detected.")
            return None

        # Extract features
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)

        return np.hstack([mfcc, chroma, centroid, zcr])

    except Exception as e:
        st.error(f"❌ Feature extraction failed: {e}")
        return None

# === Streamlit App UI ===
st.set_page_config(page_title="Voice Gender Classifier", page_icon="🎙️")
st.title("🎙️ Voice Gender Classifier")
st.markdown("Upload a voice recording and get the predicted gender using AI and signal processing.")

uploaded_file = st.file_uploader(
    "🎧 Upload an audio file",
    type=["wav", "mp3", "m4a", "aac", "flac", "ogg"]
)

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("🔍 Processing audio..."):
        features = extract_features(uploaded_file)

    if features is not None and model is not None:
        prediction = model.predict([features])[0]
        st.success(f"🧠 Predicted Gender: **{prediction.capitalize()}**")
