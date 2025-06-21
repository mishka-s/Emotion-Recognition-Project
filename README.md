# Speech Emotion Classification Using CNN + BiLSTM 

This project aims to **automatically detect emotions** from audio speech using deep learning. Given a `.wav` audio file of someone speaking, the trained model predicts their emotional state, such as **happy**, **sad**, **angry**, etc.

We use a hybrid model combining **Convolutional Neural Networks (CNN)** for spatial feature extraction and **Bidirectional LSTMs (BiLSTM)** for capturing temporal dependencies, with advanced training techniques like **data augmentation** and **cosine annealing learning rate scheduling**.

---

## 📌 Problem Statement

Understanding emotional context from speech is crucial in fields like:
- Human-Computer Interaction (HCI)
- Mental health monitoring
- Smart assistants (e.g., Alexa, Siri)
- Call center analytics

This project uses machine learning to build a robust system that classifies **spoken audio into emotions** with high accuracy.

---

## 🗃️ Dataset – RAVDESS

**RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** is a professionally recorded dataset containing:
- **7356 audio files** (speech-only subset used here)
- **24 actors** (12 male, 12 female)
- **8 emotions**: calm, happy, sad, angry, fearful, disgust, surprised, neutral
- **Format**: Mono WAV files at 48kHz

---

## Preprocessing & Feature Extraction

To prepare the audio for modeling, we perform:

### Step 1: Resampling
- All audio is converted to **22,050 Hz** using `librosa`.

### Step 2: Feature Extraction (MFCC)
- Extract **40 MFCC (Mel-Frequency Cepstral Coefficients)** from each audio.
- MFCCs are padded/truncated to a uniform shape of **(174, 40)** for input to the model.

### Step 3: Data Augmentation
Each training sample is expanded using:
- **Noise injection**: Simulates real-world noise
- **Pitch shifting**: Alters tone to simulate variability in voice
- **Speed variation**: Simulates different speech speeds

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

- **Optimizer**: Adam
- **Learning rate schedule**: CosineDecayRestarts
- **Loss function**: Categorical Crossentropy
- **Batch size**: 32
- **Epochs**: 40
- **Callbacks**: ModelCheckpoint (best `val_accuracy`)

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


