from flask import Flask, request, render_template, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models
from flask_cors import CORS

# Disable GPU + suppress warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize Flask app
app = Flask(__name__)
CORS(app) # Enable CORS for cross-origin requests from Streamlit

# Load model
# Define the base model (Xception without the top classification layer)
base_model = Xception(
    include_top=False,
    weights=None, # We will load custom weights
    input_shape=(224, 224, 3), # Input image dimensions (height, width, channels)
    pooling='avg' # Global average pooling to reduce dimensions
)

# Define the full model architecture
model = models.Sequential([
    base_model, # The Xception base
    layers.Dense(1024, activation='relu'), # A dense layer with ReLU activation
    layers.Dense(1, activation='sigmoid') # Output layer for binary classification (Real/Fake) with sigmoid activation
])

# Define the path to the model weights file
model_weights_path = os.path.join('models', 'model.weights.h5')

# Check if the model weights file exists before attempting to load
if os.path.exists(model_weights_path):
    model.load_weights(model_weights_path)
else:
    print(f"Warning: Model weights file not found at {model_weights_path}. Model will use random weights.")
    print("Please ensure 'model.weights.h5' is in a 'models' directory relative to app.py.")


# Frame extraction function
def extract_frames_from_video(input_video_path, num_frames=30):
    """
    Extracts a specified number of frames from a video file.

    Args:
        input_video_path (str): The path to the input video file.
        num_frames (int): The target number of frames to extract.

    Returns:
        numpy.ndarray: An array of processed video frames.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return np.array([]) # Return empty array if video cannot be opened

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame step to get approximately num_frames evenly distributed
    if total_frames == 0:
        print("Warning: Video has no frames.")
        return np.array([])

    frame_step = total_frames // num_frames if total_frames > num_frames else 1

    frames = []
    frame_count = 0
    success, frame = cap.read() # Read the first frame

    # Loop through frames, extracting at calculated intervals
    while success:
        if frame_count % frame_step == 0:
            # Resize frame to model input size (224x224)
            frame_resized = cv2.resize(frame, (224, 224))
            # Normalize pixel values to be between 0 and 1
            frame_normalized = frame_resized / 255.0
            frames.append(frame_normalized)
        
        success, frame = cap.read() # Read the next frame
        frame_count += 1

    cap.release() # Release the video capture object
    return np.array(frames) # Convert list of frames to a numpy array


# Classification logic
def classify_video(input_video_path, model, num_frames=30):
    """
    Classifies a video as 'Real' or 'Fake' and returns a confidence score.

    Args:
        input_video_path (str): The path to the video file.
        model (tf.keras.Model): The loaded TensorFlow/Keras model.
        num_frames (int): The number of frames to extract for analysis.

    Returns:
        float: The average prediction score (confidence) from the model,
               where values closer to 1.0 indicate 'Fake' and closer to 0.0 indicate 'Real'.
    """
    frames = extract_frames_from_video(input_video_path, num_frames)

    if frames.size == 0:
        print("No frames extracted, returning default prediction.")
        return 0.5 # Return a neutral score if no frames were processed

    predictions = []

    print(f"Extracted {len(frames)} frames for prediction.")

    # Predict on each extracted frame
    for frame in frames:
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension (1, 224, 224, 3)
        prediction = model.predict(frame, verbose=0)  # Predict each frame, suppress verbose output
        predictions.append(prediction)

    predictions = np.array(predictions)
    avg_prediction = np.mean(predictions)  # Aggregate predictions by taking the mean

    print(f"Average prediction score: {avg_prediction}")

    # Return the average prediction score (confidence)
    return float(avg_prediction)


# Routes
@app.route('/')
def index():
    """
    Renders the main index.html page.
    This route is primarily for a traditional web interface,
    not directly used by the Streamlit app.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles video upload and deepfake prediction.

    Expects a video file in the request.files.
    If 'api=true' query parameter is present, returns a plain text confidence score.
    Otherwise, renders the result.html template with the prediction.
    """
    # Check if a file was sent in the request
    if 'file' not in request.files:
        return "No file part in request", 400

    file = request.files['file']
    # Check if the file name is empty
    if file.filename == '':
        return "No selected file", 400

    # Securely get the filename and create upload directory
    filename = secure_filename(file.filename)
    upload_dir = os.path.join('static', 'uploads')
    os.makedirs(upload_dir, exist_ok=True) # Create directory if it doesn't exist
    file_path = os.path.join(upload_dir, filename)
    file.save(file_path) # Save the uploaded file

    # Classify the video and get the raw prediction score
    prediction_score = classify_video(file_path, model)

    # Construct the URL for the uploaded video (for the HTML template)
    video_url = os.path.join('uploads', filename)

    # Check if the request is from an API client (like Streamlit)
    if request.args.get('api') == 'true':
        # Return the prediction score as a plain text string
        return str(prediction_score)

    # If not an API request, render the HTML result page
    # Convert score to "Real" or "Fake" for the HTML template
    result_text = "Fake" if prediction_score > 0.5 else "Real"
    return render_template('result.html', video_url=video_url, result=result_text)


# Run app
if __name__ == '__main__':
    # Run the Flask app in debug mode (set to False for production)
    # Host '0.0.0.0' makes it accessible externally (e.g., from Streamlit)
    # Port 5002 is specified in the Streamlit app
    # threaded=False and use_reloader=False are important for some environments
    app.run(debug=True, host='0.0.0.0', port=5002, threaded=False, use_reloader=False)

