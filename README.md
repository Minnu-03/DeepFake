# 🎥 Deepfake Video Detector

## 🧠 Introduction

With the rise of synthetic media, it's becoming increasingly difficult to differentiate between real and fake video content.  
Deepfake videos, which use AI to manipulate visual content, pose significant risks to security, misinformation, and digital privacy.

This project presents a lightweight yet powerful solution that detects deepfakes by identifying subtle inconsistencies in facial features, movements, lighting, and pixel-level artifacts.  
It combines a responsive web interface with real-time backend analysis powered by a deep learning model based on the Xception architecture.

---

## 🛠️ Tech Stack

- **Flask + Flask-RESTful** – For building the backend API and handling video uploads.
- **Streamlit** – For a fast and user-friendly web interface.
- **OpenCV** – For extracting and processing frames from uploaded videos.
- **Xception** – A deep learning model pre-trained and fine-tuned for frame-by-frame analysis to detect deepfakes.

---

## 🧠 Model & Dataset

This project uses a deep learning pipeline powered by the **Xception** architecture and trained on the **UADFV** dataset.

### 🔸 Model
- **Xception** is a deep convolutional neural network known for its performance in image classification.
- It was fine-tuned for deepfake detection by analyzing individual video frames.
- The model detects subtle signs of manipulation, including:
  - Unnatural facial movements
  - Lighting inconsistencies
  - Lack of imperfections or facial asymmetry

📎 **Download Trained Model **: https://drive.google.com/drive/folders/17REF5E-DvYx3H_g_ZvTOLS8DX5-q0YIT?usp=drive_link

### 🔸 Dataset
- **UADFV Dataset**: Contains 98 videos (49 real + 49 deepfakes).
- Videos feature people speaking to the camera, with deepfakes generated using facial reenactment.

---

## 🚀 How It Works

This deepfake detection system follows a structured video analysis pipeline:

- 🎥 **User Upload**: A video file is uploaded through the Streamlit web interface.
- 📤 **API Request**: The video is sent to the Flask backend via an API call.
- 🧩 **Frame Extraction**: Using OpenCV, approximately 30 evenly spaced frames are extracted from the video.
- 🔍 **Prediction**: The frames are passed through a pre-trained Xception model to identify deepfake characteristics.
- 📊 **Score Aggregation**: Predictions from all frames are averaged to calculate a final confidence score.
- ✅ **Result Display**: Streamlit displays the result — "Real" or "Fake" — along with a detailed confidence percentage.

> A score closer to `1.0` indicates **FAKE**, while closer to `0.0` indicates **REAL**.


---

## 📹 Demo


https://github.com/user-attachments/assets/0cd8d36d-2419-405c-8719-4b9d1f2f0ad1


---

## 🙋‍♀️ Contact

- 🔗 GitHub: [Minnu-03](https://github.com/Minnu-03)
