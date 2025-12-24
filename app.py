import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

st.title("AI Face Detection Tool")

# Load cascade safely
cascade_path = "haarcascade_frontalface_default.xml"

if not os.path.exists(cascade_path):
    st.error("Cascade file not found!")
    st.stop()

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    st.error("Failed to load face detection model.")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(
            img_array,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    st.image(img_array, caption="Detected Faces", use_column_width=True)
    st.write("Number of faces detected:", len(faces))
