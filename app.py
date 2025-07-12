import streamlit as st
import numpy as np
import librosa
import joblib
import io
from pydub import AudioSegment
import os

# Point to ffmpeg (needed on Streamlit Cloud)
AudioSegment.converter = "/usr/bin/ffmpeg"

# Load Trained Model
@st.cache_resource
def load_model():
    model_path = "models/voice_gender_model.pkl"
    if not os.path.exists(model_path):
        st.error("Model file not found.")
        return None
    return joblib.load(model_path)

model = load_model()

# Extract Audio Features
def extract_features(uploaded_file):
    try:
        # Load uploaded audio and convert to mono
        audio = AudioSegment.from_file(uploaded_file)
        if audio.channels != 1:
            st.info("🔁 Converting stereo audio to mono...")
            audio = audio.set_channels(1)

        # Export to WAV in memory
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # Load with librosa
        y, sr = librosa.load(wav_io, sr=22050)
        duration = librosa.get_duration(y=y, sr=sr)
        st.write(f"📏 Audio Duration: {duration:.2f} seconds")

        if duration < 0.5:
            st.warning("⚠️ Audio too short (< 0.5s). Try a longer recording.")
            return None

        if y.size == 0:
            st.warning("⚠️ Audio contains no valid signal.")
            return None

        Extract features
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)

        return np.hstack([mfcc, chroma, centroid, zcr])

    except Exception as e:
        st.error(f"❌ Feature extraction failed: {e}")
        return None

# Streamlit Interface
st.set_page_config(page_title="Voice Gender Classifier", page_icon="🎙️")
st.title("🎙️ Voice Gender Classifier")
st.markdown("Upload a voice recording (`.wav`, `.mp3`, `.m4a`, `.ogg`, etc.) to predict the gender using AI and signal processing.")

uploaded_file = st.file_uploader("🎧 Upload an audio file", type=["wav", "mp3", "m4a", "flac", "aac", "ogg"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')

    with st.spinner("🔍 Analyzing audio..."):
        features = extract_features(uploaded_file)

    if features is not None and model is not None:
        prediction = model.predict([features])[0]
        st.success(f"🧠 Predicted Gender: **{prediction.capitalize()}**")


st.markdown("Upload a voice recording ("MODELLED BY: EMMANUEL EJIFOLABI")

