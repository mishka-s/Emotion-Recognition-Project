# streamlit_app.py

import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import io

# Load model and encoder once
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model("best_model_val_accuracy.keras")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, le = load_model_and_encoder()

# Extract MFCC function
def extract_features(file, sr=22050, n_mfcc=40, max_len=174):
    # Read bytes and decode as WAV
    audio_bytes = file.read()
    audio_stream = io.BytesIO(audio_bytes)

    # Now load using librosa
    audio, sample_rate = librosa.load(audio_stream, sr=sr)
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc).T
    if mfcc.shape[0] < max_len:
        mfcc = np.pad(mfcc, ((0, max_len - mfcc.shape[0]), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len]
    return mfcc

# App UI
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a `.wav` file of speech and this app will predict the emotion!")

uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    # Plot waveform
    audio, sr = librosa.load(uploaded_file, sr=22050)
    st.subheader("Waveform")
    fig, ax = plt.subplots()
    ax.plot(audio)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)
    
    # Feature extraction
    mfcc = extract_features(uploaded_file)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Add batch and channel dims

    # Prediction
    prediction = model.predict(mfcc)
    predicted_class = np.argmax(prediction, axis=1)[0]
    emotion = le.inverse_transform([predicted_class])[0]

    st.success(f"🧠 Predicted Emotion: **{emotion.upper()}**")

