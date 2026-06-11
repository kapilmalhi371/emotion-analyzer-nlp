import streamlit as st
from deepface import DeepFace
from transformers import pipeline
from PIL import Image
import numpy as np

# Set Streamlit page config
st.set_page_config(page_title="Multimodal Emotion Detector", layout="centered")

st.title("🎭 Multimodal Emotion Detector")
st.markdown("**Detect emotion from Text and Face Image.**")

# Cache the text model
@st.cache_resource
def load_text_model():
    return pipeline("text-classification", model="nateraw/bert-base-uncased-emotion")

# 1. Text Emotion Detection
st.header("1️⃣ Text Emotion Detection")
text_input = st.text_input("Enter a sentence:")
if st.button("Analyze Text"):
    if text_input.strip():
        with st.spinner("Analyzing text..."):
            classifier = load_text_model()
            result = classifier(text_input)[0]
        st.success(f"**Emotion:** {result['label']} ({round(result['score'] * 100, 2)}%)")
    else:
        st.warning("Please enter a sentence.")

# 2. Face Emotion Detection
st.header("2️⃣ Face Emotion Detection")
image_file = st.file_uploader("Upload a face image (JPG/PNG):", type=["jpg", "jpeg", "png"])
if image_file is not None:
    try:
        with st.spinner("Analyzing face image..."):
            img_pil = Image.open(image_file).convert("RGB")
            img_np = np.array(img_pil)
            img_bgr = img_np[:, :, ::-1]  # RGB to BGR for OpenCV

            result = DeepFace.analyze(img_path=img_bgr, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']

        st.image(img_pil, caption="Uploaded Image", use_container_width=True)
        st.success(f"**Detected Emotion from Face:** {dominant_emotion}")
    except Exception as e:
        st.error(f"Face emotion detection failed: {str(e)}")
