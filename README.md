# Speech Emotion Classification Using CNN + BiLSTM 

This project aims to **automatically detect emotions** from audio speech using deep learning. Given a `.wav` audio file of someone speaking, the trained model predicts their emotional state, such as **happy**, **sad**, **angry**, etc.

We use a hybrid model combining **Convolutional Neural Networks (CNN)** for spatial feature extraction and **Bidirectional LSTMs (BiLSTM)** for capturing temporal dependencies, with advanced training techniques like **data augmentation** and **cosine annealing learning rate scheduling**.

---

## Problem Statement

Understanding emotional context from speech is crucial in fields like:
- Human-Computer Interaction (HCI)
- Mental health monitoring
- Smart assistants (e.g., Alexa, Siri)
- Call center analytics

This project uses machine learning to build a robust system that classifies **spoken audio into emotions** with high accuracy.

---
## Web App Usage:

You can try the working web app here:

**https://mishka-singla-emotion-recognition.streamlit.app/**

### How to use:

- Upload any .wav file (like from RAVDESS)

- Wait for the model to process it

- View the predicted emotion and waveform

---

You can use the provided test file: sample_audio/03-01-03-01-01-01-03.wav
(Which corresponds to "Happy")
## Dataset – RAVDESS

**RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** (https://zenodo.org/record/1188976) is a professionally recorded dataset containing:
- **7356 audio files** (speech-only subset used here)
- **24 actors** (12 male, 12 female)
- **8 emotions**: calm, happy, sad, angry, fearful, disgust, surprised, neutral
- **Format**: Mono WAV files at 48kHz

---

## Preprocessing & Feature Extraction

To prepare the audio for modeling, we perform:

### Step 1: Resampling
- All audio is converted to **22,050 Hz** using `librosa`. This ensures uniformity and reduces computational load.

### Step 2: Feature Extraction (MFCC)
- Extract **40 MFCC (Mel-Frequency Cepstral Coefficients)** from each audio.
- MFCCs are padded/truncated to 174 time steps, giving a uniform shape of **(174, 40)** for input to the model.
- MFCCs capture timbre and spectral shape, which reflect emotion better than raw waveforms.

### Step 3: Data Augmentation
Each training sample is expanded using:
- **Noise injection**: Simulates real-world background noise and adds Gaussian noise to MFCCs
- **Pitch shifting**: Alters tone to simulate variability in voice and uses np.roll on the MFCC time axis
- **Speed variation**: Simulates different speech speeds. It helps in time-stretching and zero-padding

These augmentations help improve model generalization and performance.

---

## Model Architecture

The model is designed to combine the benefits of:
- **CNN**: For extracting spatial features from MFCCs
- **BiLSTM**: For capturing the sequence (temporal) relationships in speech 
- **Dropout** and **Batch Normalization**: To prevent overfitting and stabilize training

## Model Pipeline

| Layer              | Details                              |
|-------------------|---------------------------------------|
| Input             | Shape: (174, 40, 1)                   |
| Conv2D            | 32 filters, 3x3, ReLU + BatchNorm     |
| MaxPooling2D      | 2x2                                   |
| Dropout           | 0.3                                   |
| Conv2D            | 64 filters, 3x3, ReLU + BatchNorm     |
| MaxPooling2D      | 2x2                                   |
| Dropout           | 0.3                                   |
| Reshape           | For LSTM compatibility                |
| BiLSTM            | 64 units (bidirectional)              |
| Dropout           | 0.5                                   |
| Dense             | Softmax activation                    |

---

## Training Strategy

- **Loss**: Categorical Crossentropy
- **Optimizer**: Adam
- **Learning Rate**: CosineDecayRestarts (dynamic LR scheduler)
- **Validation split**: 20% of data
- **Batch Size**: 32
- **Epochs**: 40
- **Checkpointing**: Best model saved using `val_accuracy` as metric

---

## Evaluation Metrics

After training, the model is evaluated using:

| Metric                  | Description                         |
|--------------------------|-------------------------------------|
| **Validation Accuracy**  | Measures final accuracy on val set |
| **Confusion Matrix**     | Shows where the model misclassifies |
| **Classification Report**| Precision, recall, F1-score        |
| **Per-Class Accuracy**   | Class-wise performance tracking     |

### The following conditions are duly satisfied:

- F1 score should be greater than 80%- **0.83**

- Accuracy of each class should be greater than 75%- **Accuracy table printed below**
   
- Overall accuracy should be greater than 80%- **82.89%**

![image](https://github.com/user-attachments/assets/f78679f4-3683-46c1-9221-05f716074cfa)

![image](https://github.com/user-attachments/assets/4313aeff-5a63-47ea-a76e-a4127922f420)

---

## How to Test the Model on a New Audio File

You can use the provided `test_model.py` script to classify the emotion from any `.wav` speech file.

---

### Requirements

Before running the test script, ensure the following files are present in your project folder:

| File                            | Purpose                                      |
|---------------------------------|----------------------------------------------|
| `test_model.py`                | Script to predict emotion from `.wav` file   |
| `best_model_val_accuracy.keras`| Trained model saved during training          |
| `label_encoder.pkl`            | Label encoder saved using `joblib.dump()`    |
| `*.wav`                        | A test speech file (e.g., from RAVDESS)      |

---
### Use this command:

python test_model.py path/to/audio.wav

### This will output:

Predicted Emotion: {emotion}



