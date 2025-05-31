import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import pickle

fruit_veg_model = tf.keras.models.load_model("fruit_vs_vegetable_model.keras")
fruit_model = tf.keras.models.load_model("fruit_quality_model.keras")
veg_model = tf.keras.models.load_model("vegetable_quality_model.keras")

with open("fruit_label_encoder.pkl", "rb") as f:
    fruit_le = pickle.load(f)

veg_classes = ['fresh', 'stale']

st.title("🥕 Fruit & Vegetable Quality Detector")
st.markdown("Upload an image or take a photo to classify.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  

    fv_pred = fruit_veg_model.predict(img_array)
    fv_class = np.argmax(fv_pred)
    fv_label = "fruit" if fv_class == 0 else "vegetable"
    st.subheader(f"🧠 Detected Type: **{fv_label.capitalize()}**")

    if fv_label == "fruit":
        quality_pred = fruit_model.predict(img_array)
        pred_idx = np.argmax(quality_pred)
        label = fruit_le.inverse_transform([pred_idx])[0]
        confidence = np.max(quality_pred) * 100
    else:
        quality_pred = veg_model.predict(img_array)
        pred_idx = np.argmax(quality_pred)
        label = veg_classes[pred_idx]
        confidence = np.max(quality_pred) * 100
        
    st.success(f"🍽️ Quality: **{label.upper()}** ({confidence:.2f}%)")
