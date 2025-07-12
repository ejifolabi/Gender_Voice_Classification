import streamlit as st
import librosa
import numpy as np
import joblib
from pydub import AudioSegment
import io
import os

# For Streamlit Cloud: point to ffmpeg
AudioSegment.converter = "/usr/bin/ffmpeg"

# === Load Trained Model ===
@st.cache_resource
def load_model():
    return joblib.load("models/voice_gender_model.pkl")

model = load_model()

# === Feature Extraction ===
def extract_features(file):
    try:
        # Convert to WAV using pydub
        audio = AudioSegment.from_file(file)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # Load audio with librosa
        y, sr = librosa.load(wav_io, sr=22050, mono=True)

        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)

        return np.hstack([mfcc, chroma, centroid, zcr])
    except Exception as e:
        st.error(f"❌ Feature extraction failed: {e}")
        return None

# === UI ===
st.set_page_config(page_title="Voice Gender Classifier", page_icon="🎙️")
st.title("🎙️ Voice Gender Classifier")
st.markdown("Upload an audio file (`.wav`, `.mp3`, `.aac`, `.m4a`, etc.) and I’ll predict the gender using AI and signal processing.")

uploaded_file = st.file_uploader("🎧 Upload an audio file", type=["wav", "mp3", "ogg", "flac", "m4a", "aac"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with st.spinner("🔍 Extracting features..."):
        features = extract_features(uploaded_file)

    if features is not None:
        prediction = model.predict([features])[0]
        st.success(f"🧠 Predicted Gender: **{prediction.capitalize()}**")

st.markdown("---")
st.markdown("👨‍💻 **Modelled By: Emmanuel Ejifolabi**")
