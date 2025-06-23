import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os

# Load model
model = tf.keras.models.load_model("asl_model.h5")

# Class labels (alphabet A-Z)
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['del', 'nothing', 'space']

# Streamlit UI
st.title("🖐️ ASL Detection Web App")
st.write("Upload an image of an ASL gesture and the model will predict the letter.")

# Image upload
uploaded_file = st.file_uploader("Choose an ASL image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = image.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    pred = model.predict(img_array)
    predicted_class = classes[np.argmax(pred)]

    st.success(f"✅ Predicted Letter: **{predicted_class}**")
