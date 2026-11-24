"""Streamlit frontend: upload an image, send it to the Flask inference
server (serve_model.py), and show the extracted JSON next to the image."""
import base64
import io

import requests
import streamlit as st
from PIL import Image

st.set_page_config(layout="wide")

FLASK_SERVER_URL = "http://localhost:5000/process_image"

st.title("Image to JSON Extractor")
st.write("Upload an image and the model will extract structured JSON data.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    with st.spinner("Processing image..."):
        response = requests.post(FLASK_SERVER_URL, json={"image": image_base64})

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image")
        with col2:
            if response.status_code == 200:
                st.subheader("Extracted JSON Data")
                st.json(response.json())
            else:
                st.error(f"Error: {response.status_code}")
