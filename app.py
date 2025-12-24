import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("AI Face Detection Tool")

# Load model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)

    st.image(img_array, caption="Detected Faces", use_column_width=True)
    st.write("Number of faces detected:", len(faces))
