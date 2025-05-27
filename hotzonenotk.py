import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Thermal Simulation and Hot Zone Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Apply thermal colormap
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Detect hot zones (threshold > 200)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    highlight = image_cv.copy()
    highlight[mask == 255] = [0, 0, 255]  # Red overlay

    # Display all images
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    st.subheader("Thermal Colormap")
    st.image(cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.subheader("Hot Zones Detected")
    st.image(cv2.cvtColor(highlight, cv2.COLOR_BGR2RGB), use_column_width=True)
