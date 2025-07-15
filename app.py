import os
import librosa
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa.display
import joblib
import soundfile as sf
from pydub import AudioSegment
import tempfile

# === CONFIG ===
SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # seconds
MODEL_PATH = "models/voice_gender_model.pkl"

# === Feature Extraction ===
def extract_features(y, sr=16000):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    features = np.hstack([
        mfcc.mean(axis=1),
        chroma.mean(axis=1),
        contrast.mean(axis=1),
        zcr.mean(),
        rms.mean()
    ])
    return features

# === Prediction Functions ===
def predict_single_segment(y, model):
    y = librosa.util.fix_length(y, size=CHUNK_DURATION * SAMPLE_RATE)
    features = extract_features(y).reshape(1, -1)
    prediction = model.predict(features)[0]
    return prediction

def predict_from_first_chunk(y, model):
    if len(y) < CHUNK_DURATION * SAMPLE_RATE:
        return "Audio too short (min 3s)", None
    y = y[:CHUNK_DURATION * SAMPLE_RATE]
    label = predict_single_segment(y, model)
    return label, y

def predict_with_sliding_window(y, model):
    if len(y) < CHUNK_DURATION * SAMPLE_RATE:
        return "Audio too short (min 3s)", None
    step = CHUNK_DURATION * SAMPLE_RATE
    results = []
    for i in range(0, len(y), step):
        chunk = y[i:i+step]
        if len(chunk) < step:
            continue
        chunk = librosa.util.fix_length(chunk, size=step)
        pred = predict_single_segment(chunk, model)
        results.append(pred)
    if not results:
        return "No valid chunks found", None
    final_pred = max(set(results), key=results.count)
    return final_pred, y

# === Visualization ===
def display_audio_visuals(y, sr):
    st.subheader("Waveform")
    plt.figure(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr)
    st.pyplot(plt)

    st.subheader("Spectrogram")
    plt.figure(figsize=(10, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    st.pyplot(plt)

# === Streamlit UI ===
st.set_page_config(page_title="Voice Gender Recognition", layout="centered")
st.title("🎤 Voice Gender Recognition")
st.markdown("Upload a voice sample (≥3 seconds) to predict the speaker's gender.")

uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "flac", "ogg", "aac"])
analysis_mode = st.radio("Choose Prediction Mode", ["Use First 3 Seconds", "Full Audio Analysis"])

if uploaded_file:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            audio = AudioSegment.from_file(uploaded_file)
            audio.export(temp_file.name, format="wav")
            temp_path = temp_file.name

        y, sr = librosa.load(temp_path, sr=SAMPLE_RATE)

        if len(y) < CHUNK_DURATION * sr:
            st.warning("Audio too short. Must be at least 3 seconds.")
        else:
            model = joblib.load(MODEL_PATH)

            if analysis_mode == "Use First 3 Seconds":
                label, audio_used = predict_from_first_chunk(y, model)
            else:
                label, audio_used = predict_with_sliding_window(y, model)

            if isinstance(label, str):
                st.error(label)
            else:
                result = "👩 Female" if label == 1 else "👨 Male"
                st.success(f"**Predicted Gender: {result}**")
                st.audio(temp_path, format='audio/wav')
                display_audio_visuals(audio_used, SAMPLE_RATE)

    except Exception as e:
        st.error(f"Failed to process audio: {e}")

st.markdown("---")
st.markdown("DEVELOPED BY: EMMANUEL EJIFOLABI")
