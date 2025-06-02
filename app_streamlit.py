import streamlit as st
import requests
from time import sleep

FLASK_API_URL = "http://127.0.0.1:5002/predict?api=true"

# Setting Streamlit page configuration
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🎥", 
    layout="centered", 
    initial_sidebar_state="auto"
)

# --- Header Section ---
st.title("🎥 Deepfake Video Detector")
st.caption("Check if a video is real or AI-generated.")

# --- Instructions ---
st.header("How to Use:")
st.markdown("""
1.  **Upload a video file** below. Supported formats: `.mp4`, `.mov`, `.avi`.
2.  Click the **Submit** button to start the analysis.
3.  **View the prediction result** and confidence score.
""")

# --- Video Upload Section ---
st.subheader("Upload Your Video")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"], help="Max file size is typically limited by Streamlit and server settings.")

if uploaded_file is not None:
    st.markdown("---")
    st.subheader("Preview Your Uploaded Video:")
    st.video(uploaded_file)
    st.markdown("---")

    if st.button('Submit for Analysis', help="Click to send your video for deepfake detection."):
        file_bytes = uploaded_file.read()
        files = {'file': (uploaded_file.name, file_bytes, uploaded_file.type)}

        # --- Processing Feedback ---
        with st.spinner('Analyzing video... This might take a few moments depending on video length and server load.'):
            # Attempt to connect to Flask app multiple times
            for attempt in range(1, 6): # 5 retries
                try:
                    response = requests.post(FLASK_API_URL, files=files, timeout=600) # Increased timeout
                    if response.status_code == 200:
                        try:
                            confidence_score = float(response.text)
                            prediction_label = "Fake" if confidence_score > 0.5 else "Real"

                            st.success(f"### Prediction Result: **{prediction_label.upper()}**")

                            # Display confidence with conditional coloring
                            if prediction_label == "Fake":
                                st.error(f"**Confidence:** :red[{confidence_score:.2%}] (Highly likely to be AI-generated)")
                            else:
                                st.success(f"**Confidence:** :green[{1 - confidence_score:.2%}] (Highly likely to be real)") # Display confidence for 'Real' directly

                            st.info(f"*(A score closer to 1.0 indicates 'Fake', closer to 0.0 indicates 'Real'. Your video scored {confidence_score:.2f})*")

                            # Add a general note about indicators (not specific to this model's output)
                            st.markdown(
                                """
                                **Note on Indicators:** Advanced deepfake detection often looks for subtle inconsistencies such as:
                                * Unnatural eye movements or blinks
                                * Inconsistent lighting on the face or body
                                * Pixelation or blurring around the edges of a manipulated subject
                                * Absence of natural human imperfections
                                * Audio-visual synchronization issues
                                """
                            )
                        except ValueError:
                            st.error("Error: Could not parse prediction result from the detection server. Please try again.")
                        break # Exit retry loop on success
                    else:
                        st.error(f"Analysis failed. Server responded with status code: {response.status_code}. Please try again later.")
                        break # Exit retry loop on server error
                except requests.exceptions.ConnectionError:
                    if attempt < 5:
                        st.warning(f"Connection to detection server failed. Retrying... (Attempt {attempt}/5)")
                        sleep(2) # Wait before retrying
                    else:
                        st.error("Failed to connect to the deepfake detection server after multiple attempts. Please ensure the server is running and accessible.")
                except requests.exceptions.Timeout:
                    st.error("The detection process timed out. The video might be too long or the server is overloaded. Please try a shorter video or try again later.")
                except Exception as e:
                    st.error(f"An unexpected error occurred during analysis: {e}")
                    break # Exit retry loop on other errors
            else:
                st.error("Failed to process the video. Please check your network connection and server status.")

# --- Footer/Branding/Contact Section ---
st.markdown("---")
st.subheader("About This Detector")
st.markdown(
    """
    This Deepfake Detector leverages a deep learning model,
    specifically a pre-trained Xception network.
    The model is trained to distinguish between real and AI-generated (deepfake) videos
    by identifying subtle anomalies often present in manipulated media.
    """
)

st.subheader("Disclaimer")
st.warning(
    """
    **Important Note:** No deepfake detection technology is 100% accurate.
    Results should be considered as an indicator and not definitive proof.
    Deepfake technology is constantly evolving, and detection methods are also under continuous development.
    """
)

st.subheader("Contact Us")
st.markdown(
    """
    For questions, feedback, or collaborations, please feel free to reach out:
    * **Email:** [minnusrithumnoori03@gmail.com](mailto:support@yourcompany.com)
    * **Website:** www.team2.com
    * **GitHub:** https://github.com/Minnu-03 
    """
)

# Optional: Add a simple footer
st.markdown("---")
st.markdown("© 2025 Deepfake Detector. All rights reserved.")
