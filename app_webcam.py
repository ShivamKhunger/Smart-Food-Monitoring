import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image

fruit_veg_model = tf.keras.models.load_model("fruit_vs_vegetable_model.h5")
fruit_model = tf.keras.models.load_model("final_model.h5")
veg_model = tf.keras.models.load_model("vegetable_quality_model.h5")

with open("fruit_label_encoder.pkl", "rb") as f:
    fruit_le = pickle.load(f)

veg_classes = ['fresh', 'stale']

shelf_life_map = {
    'fresh': 5, 'ripe': 3, 'overripe': 1, 'unripe': 6, 'stale': 0
}

tips = {
    "banana": {
        "unripe": ("Rich in resistant starch", "Let it ripen naturally"),
        "ripe": ("Energy booster 🍌", "Eat raw or in cereal"),
        "overripe": ("Good for digestion", "Make banana bread 🍞"),
        "fresh": ("Easy to digest", "Peel and eat directly"),
        "stale": ("Avoid eating", "Compost it")
    },
    "apple": {
        "unripe": ("Tart and firm", "Use in chutney or pie"),
        "ripe": ("Rich in antioxidants", "Eat raw or in salad 🥗"),
        "fresh": ("Full of fiber & vitamins", "Snack or dip in peanut butter"),
        "overripe": ("Soft texture", "Make applesauce or jam"),
        "stale": ("Low nutrition", "Use for vinegar or compost")
    },
    "orange": {
        "unripe": ("Sour taste", "Use for marmalade"),
        "ripe": ("High in juice content 🍊", "Juice or eat raw"),
        "fresh": ("Vitamin C rich", "Peel and eat or make juice"),
        "overripe": ("May ferment", "Use zest or peels"),
        "stale": ("Not safe to eat", "Peels for cleaning 🧼")
    }
}

st.set_page_config(layout="centered")
st.title("🍅 Real-time Food Quality Detector")
st.markdown("Detect fruit/vegetable quality + shelf life + tip + recipe in real-time.")

FRAME = st.empty()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Could not open webcam.")
    st.stop()

while True:
    ret, frame = cap.read()
    if not ret:
        st.warning("🔄 Waiting for webcam...")
        continue

    try:
        crop = cv2.resize(frame, (224, 224))
        inp = np.expand_dims(crop / 255.0, axis=0)

        fv_pred = fruit_veg_model.predict(inp)
        fv_class = np.argmax(fv_pred)
        fv_label = "fruit" if fv_class == 0 else "vegetable"

        if fv_label == "fruit":
            pred = fruit_model.predict(inp)
            label = fruit_le.inverse_transform([np.argmax(pred)])[0]
        else:
            pred = veg_model.predict(inp)
            label = veg_classes[np.argmax(pred)]

        confidence = np.max(pred) * 100
        quality = label.lower()
        shelf = shelf_life_map.get(quality, "N/A")

        
        object_name = "banana"
        nutri, recipe = tips.get(object_name, {}).get(quality, ("", ""))


        cv2.putText(frame, f"{fv_label.upper()} - {label.upper()} ({confidence:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, f"📆 Shelf Life: {shelf} days", (10, 60),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 139), 2)
        cv2.putText(frame, f"🍽 Tip: {nutri}", (10, 85),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 100, 0), 2)
        cv2.putText(frame, f"🍳 Try: {recipe}", (10, 110),
                    cv2.FONT_HERSHEY_PLAIN, 1, (139, 0, 0), 2)

    except Exception as e:
        print("❌ Prediction error:", e)
        cv2.putText(frame, "Prediction Error", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME.image(rgb)

cap.release()
