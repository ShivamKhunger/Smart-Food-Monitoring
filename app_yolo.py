import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pickle
from ultralytics import YOLO

fruit_veg_model = tf.keras.models.load_model("fruit_vs_vegetable_model.h5")
fruit_model = tf.keras.models.load_model("final_model.h5")
veg_model = tf.keras.models.load_model("vegetable_quality_model.h5")

with open("fruit_label_encoder.pkl", "rb") as f:
    fruit_le = pickle.load(f)

yolo = YOLO("yolov8n.pt")

fruit_veg_classes = ['apple', 'banana', 'orange', 'carrot', 'broccoli']
veg_classes = ['fresh', 'stale']

shelf_life_map = {
    'fresh': 5, 'ripe': 3, 'overripe': 1, 'unripe': 6, 'stale': 0
}

tips = {
    "banana": {
        "unripe": ("Rich in resistant starch", "Let it ripen at room temperature"),
        "ripe": ("Energy booster 🍌", "Eat raw or in cereal"),
        "overripe": ("Good for digestion", "Make banana bread 🍞"),
        "fresh": ("Soft texture, easy snack", "Peel and eat directly"),
        "stale": ("Avoid eating", "Compost it")
    },
    "apple": {
        "ripe": ("High in antioxidants", "Slice into salad 🥗"),
        "fresh": ("Rich in fiber", "Eat raw or dip in peanut butter"),
        "overripe": ("Can still be used", "Make applesauce"),
        "unripe": ("Tart flavor", "Use in chutney or pickle"),
        "stale": ("Low nutrition", "Make vinegar or compost")
    },
    "orange": {
        "fresh": ("Vitamin C rich 🍊", "Eat raw or make juice"),
        "ripe": ("Juicy and sweet", "Best for fresh juice"),
        "overripe": ("Slightly fermented", "Use for zest or peels"),
        "unripe": ("Sour in taste", "Use in marmalade"),
        "stale": ("Not safe to eat", "Use peel for cleaning 🧼")
    },
    "carrot": {
        "fresh": ("Great for eyesight 🥕", "Eat raw or in soup"),
        "stale": ("Avoid eating", "Use in compost")
    },
    "broccoli": {
        "fresh": ("Iron rich 🥦", "Steam or stir fry"),
        "stale": ("Not nutritious", "Discard")
    }
}

st.set_page_config(page_title="YOLO Food Quality", layout="wide")
st.title(" Real-time Food Quality Detection")
st.write("Live object detection + quality + shelf life + nutritional tips")

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("❌ Could not open webcam")
    st.stop()

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Failed to grab frame.")
        break

    results = yolo.predict(frame, conf=0.25, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = yolo.names[cls_id].lower()

        if label in fruit_veg_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            pad = 15
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)

            crop = frame[y1:y2, x1:x2]

            if crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            resized = cv2.resize(crop, (224, 224))
            input_array = np.expand_dims(resized / 255.0, axis=0)

            fv_pred = fruit_veg_model.predict(input_array)
            fv_class = np.argmax(fv_pred)
            fv_label = "fruit" if fv_class == 0 else "vegetable"

            if fv_label == "fruit":
                pred = fruit_model.predict(input_array)
                quality = fruit_le.inverse_transform([np.argmax(pred)])[0]
            else:
                pred = veg_model.predict(input_array)
                quality = veg_classes[np.argmax(pred)]

            confidence = np.max(pred) * 100
            shelf_days = shelf_life_map.get(quality.lower(), "N/A")
            nutri, recipe = tips.get(label, {}).get(quality.lower(), ("", ""))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label.upper()} | {fv_label.upper()} | {quality.upper()} ({confidence:.1f}%)",
                    (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  

            cv2.putText(frame, f"Shelf Life: {shelf_days} days", (x1, y2 + 20),
            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 139), 2)  

            if nutri:
             cv2.putText(frame, f"Tip: {nutri}", (x1, y2 + 40),
             cv2.FONT_HERSHEY_PLAIN, 1, (0, 100, 0), 2)  

            if recipe:
                cv2.putText(frame, f"Try: {recipe}", (x1, y2 + 60),
                cv2.FONT_HERSHEY_PLAIN, 1, (139, 0, 0), 2) 

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(rgb)

cap.release()
