import streamlit as st
import numpy as np
import librosa
import pickle
from sklearn.svm import SVC

# Load the trained model
model = pickle.load(open('modelForPrediction.sav', 'rb'))

# Feature extraction function
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    X, sample_rate = librosa.load(file_name, sr=None)
    if chroma:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    return result

# Streamlit app
st.title("Speech Emotion Recognition")
st.write("Upload an audio file to predict the emotion")

# File uploader
audio_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "ogg"])

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')
    with open("temp.wav", "wb") as f:
        f.write(audio_file.getbuffer())
    
    # Extract features and predict
    features = extract_feature("temp.wav").reshape(1, -1)
    prediction = model.predict(features)
    
    # Display prediction
    st.write(f"The predicted emotion is: **{prediction[0].capitalize()}**")
