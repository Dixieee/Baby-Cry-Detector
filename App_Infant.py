import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import normalize

# Load pre-trained model (update the path if necessary)
model_path = 'baby_cry_model_akhir_banget.h5'
model = load_model(model_path)

# Kelas yang ada
classes = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired', 'not_baby']

# Define the function to load and process the audio
def process_user_audio(audio_file, target_sample_rate=22050, fixed_length=66150):
    y, sr = librosa.load(audio_file, sr=target_sample_rate)
    if len(y) < fixed_length:
        y = librosa.util.fix_length(y, size=fixed_length)
    return y, sr

# Extract features from the audio
def extract_features(y, sr, n_mfcc=13, fixed_length=66150):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.mean(mfcc, axis=1)  # Take the mean of each MFCC across time frames
    return mfcc

# Define the prediction function
def predict_audio_class(model, audio_file):
    y, sr = process_user_audio(audio_file)
    features = extract_features(y, sr)
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    features_norm = normalize(features)
    pred = model.predict(features_norm)
    predicted_class_idx = np.argmax(pred)
    predicted_class = classes[predicted_class_idx]
    confidence = np.max(pred) * 100  # Get confidence percentage
    return predicted_class, confidence, pred

# Streamlit app layout
st.title("Audio Classification with Pre-trained Model")
st.write("Upload a .wav file to classify.")

# File upload widget
audio_file = st.file_uploader("Upload Audio File", type=["wav"])

# If file is uploaded
if audio_file is not None:
    # Display original audio waveform
    y, sr = process_user_audio(audio_file)
    st.audio(audio_file, format="audio/wav")
    
    # Feature extraction and prediction
    predicted_class, confidence, pred = predict_audio_class(model, audio_file)
    
    # Show predicted class and confidence
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
    
    # Display the full prediction probabilities (optional)
    st.write("Prediction Probabilities (for all classes):")
    for idx, class_name in enumerate(classes):
        st.write(f"{class_name}: {pred[0][idx] * 100:.2f}%")
    
    # Visualize the original audio waveform
    st.subheader("Original Audio Waveform")
    plt.figure(figsize=(15, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Original Audio")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    st.pyplot()

# To run the app, use: 
# streamlit run audio_classification_app.py
