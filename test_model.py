import argparse
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load model and label encoder  
model = load_model("best_model_val_accuracy.keras")
le = joblib.load("label_encoder.pkl")

# Feature extraction function
def extract_mfcc(file_path, sr=22050, n_mfcc=40, max_len=174):
    audio, sample_rate = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc).T
    if mfcc.shape[0] < max_len:
        mfcc = np.pad(mfcc, ((0, max_len - mfcc.shape[0]), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len]
    return mfcc

# Main prediction function
def predict_emotion(file_path):
    mfcc = extract_mfcc(file_path)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Reshape to (1, 174, 40, 1)
    predictions = model.predict(mfcc)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_label = le.inverse_transform([predicted_index])[0]
    print(f"\n Predicted Emotion: **{predicted_label.upper()}**")

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict emotion from a speech WAV file.")
    parser.add_argument("audio_path", type=str, help="Path to the .wav file")
    args = parser.parse_args()
    predict_emotion(args.audio_path)
