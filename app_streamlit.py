import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models

# Load model (local path)
model_save_path = 'models/model.weights.h5'
base_model = Xception(
    include_top=False,
    weights=None,
    input_shape=(224, 224, 3),
    pooling='avg'
)

model = models.Sequential([
    base_model,
    layers.Dense(1024, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.load_weights(model_save_path)

# Frame extraction function
def extract_frames_from_video(input_video_path, num_frames=30):
    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = total_frames // num_frames if total_frames > num_frames else 1

    frames = []
    frame_count = 0
    success, frame = cap.read()
    while success:
        if frame_count % frame_step == 0:
            frame_resized = cv2.resize(frame, (224, 224))
            frame_normalized = frame_resized / 255.0
            frames.append(frame_normalized)
        success, frame = cap.read()
        frame_count += 1

    cap.release()
    return np.array(frames)

# Classification logic
def classify_video(input_video_path):
    frames = extract_frames_from_video(input_video_path, num_frames=30)
    predictions = []

    for frame in frames:
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension
        prediction = model.predict(frame, verbose=0)  # Predict each frame
        predictions.append(prediction)

    predictions = np.array(predictions)
    avg_prediction = np.mean(predictions)  # Aggregate predictions
    return "Fake" if avg_prediction > 0.5 else "Real"

# Streamlit UI
st.title('Deepfake Video Detection')
st.markdown('Upload a video and check if it is real or fake.')

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the file temporarily
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    
    result = classify_video("uploaded_video.mp4")
    st.video(uploaded_file)
    st.write(f"Prediction: {result}")
