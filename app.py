from flask import Flask, request, render_template, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models

# Disable GPU + suppress warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize Flask app
app = Flask(__name__)

# Load model
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

model_weights_path = os.path.join('models', 'model.weights.h5')
model.load_weights(model_weights_path)

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
def classify_video(input_video_path, model, num_frames=30):
    frames = extract_frames_from_video(input_video_path, num_frames)
    predictions = []

    for frame in frames:
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension
        prediction = model.predict(frame, verbose=0)  # Predict each frame
        predictions.append(prediction)

    predictions = np.array(predictions)
    avg_prediction = np.mean(predictions)  # Aggregate predictions

    return "Fake" if avg_prediction > 0.5 else "Real"

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save uploaded file
    filename = secure_filename(file.filename)
    upload_dir = os.path.join('static', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)
    file.save(file_path)

    # Classify video
    result = classify_video(file_path, model)

    # Render result template
    return render_template('result.html', video_path=file_path, result=result)


# Run app
if __name__ == '__main__':
    app.run(debug=True)
