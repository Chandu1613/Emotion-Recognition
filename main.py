import streamlit as st
import tempfile
import os
import numpy as np
import tensorflow as tf
import joblib
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from preprocessing import AudioFeatureExtractor


# ---------------------------------------------------------------------
# Load models and artifacts (default paths)
# ---------------------------------------------------------------------
cnn_model = tf.keras.models.load_model("models/cnn_stage1_model.keras", compile=False)
hybrid_model = tf.keras.models.load_model("models/hybrid_ser_model.keras", compile=False)

spec_mean = np.load("artifacts/spec_mean.npy")
spec_std = np.load("artifacts/spec_std.npy")
scaler = joblib.load("artifacts/pros_scaler.pkl")
label_encoder = joblib.load("artifacts/label_encoder.pkl")

extractor = AudioFeatureExtractor()


# ---------------------------------------------------------------------
# Helper: predict emotion
# ---------------------------------------------------------------------
def predict_emotion(filepath, model_choice="hybrid"):
    spec = extractor.extract_spectrogram_features(filepath, apply_aug=False)
    pros = extractor.extract_prosodic(filepath)

    # normalize
    spec_norm = (spec - spec_mean) / (spec_std + 1e-8)

    # Fix input shape for CNN/hybrid
    if spec_norm.ndim == 3:  # (128, 200, 3)
        spec_input = np.expand_dims(spec_norm, axis=0)  # (1, 128, 200, 3)
    elif spec_norm.ndim == 4:
        spec_input = spec_norm
    else:
        raise ValueError(f"Unexpected spectrogram shape: {spec_norm.shape}")

    pros_input = scaler.transform([pros])

    # predict
    if model_choice == "cnn":
        probs = cnn_model.predict(spec_input)[0]
    else:
        probs = hybrid_model.predict([spec_input, pros_input])[0]

    classes = list(label_encoder.classes_)
    top_idx = np.argsort(probs)[::-1]
    return probs, classes, top_idx



# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="Speech Emotion Recognition", layout="wide")
st.title("🎤 Speech Emotion Recognition")

input_mode = st.radio("Choose input mode:", ["Upload WAV", "Record with Microphone"])


# ------------------------------
# Option 1: Upload WAV
# ------------------------------
if input_mode == "Upload WAV":
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_path = tmp.name
            tmp.write(uploaded_file.read())

        st.audio(temp_path)

        probs, classes, top_idx = predict_emotion(temp_path)
        st.success(f"**Predicted emotion:** {classes[top_idx[0]]} ({probs[top_idx[0]]:.3f})")

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh([classes[i] for i in top_idx], probs[top_idx])
        ax.invert_yaxis()
        ax.set_xlabel("Probability")
        st.pyplot(fig)

        os.remove(temp_path)  # cleanup


# ------------------------------
# Option 2: Record via Microphone
# ------------------------------
elif input_mode == "Record with Microphone":
    sr = 16000
    duration = st.slider("Recording duration (seconds)", 2, 10, 5)

    if st.button("Start Recording"):
        st.info("Recording... Speak now!")
        recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_path = tmp.name
        sf.write(temp_path, recording, sr)

        st.audio(temp_path)

        # Predict
        probs, classes, top_idx = predict_emotion(temp_path)
        st.success(f"**Predicted emotion:** {classes[top_idx[0]]} ({probs[top_idx[0]]:.3f})")

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh([classes[i] for i in top_idx], probs[top_idx])
        ax.invert_yaxis()
        ax.set_xlabel("Probability")
        st.pyplot(fig)

        os.remove(temp_path)