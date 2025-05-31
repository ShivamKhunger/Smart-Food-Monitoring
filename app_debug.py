import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import numpy as np
import tensorflow as tf
import pickle
import cv2

fruit_veg_model = tf.keras.models.load_model("fruit_vs_vegetable_model.h5")
fruit_model = tf.keras.models.load_model("final_model.h5")
veg_model = tf.keras.models.load_model("vegetable_quality_model.h5")

with open("fruit_label_encoder.pkl", "rb") as f:
    fruit_le = pickle.load(f)

veg_classes = ['fresh', 'stale']

shelf_life_map = {
    'fresh': 5,
    'ripe': 3,
    'overripe': 1,
    'unripe': 6,
    'stale': 0
}

tips = {
    "banana": {
        "ripe": ("Great energy booster 🍌", "Use in smoothie or cereal"),
        "overripe": ("Good for digestion", "Make banana bread 🍞"),
        "stale": ("Avoid eating", "Compost it")
    },
    "apple": {
        "fresh": ("Rich in fiber & vitamins", "Eat raw or in salad 🥗"),
        "stale": ("Low nutrition", "Make jam or cider")
    },
    "orange": {
        "fresh": ("High in Vitamin C 🍊", "Juice or raw"),
        "stale": ("Not safe to eat", "Use peel for natural cleaner")
    },
    "grape": {
        "fresh": ("Packed with antioxidants 🍇", "Use in fruit salad"),
        "stale": ("Avoid eating", "Discard")
    },
    "mango": {
        "ripe": ("High in Vitamin A 🥭", "Make mango shake"),
        "overripe": ("Sweet and juicy", "Make aamras or ice cream"),
        "stale": ("Not edible", "Throw away")
    }
}

st.title("🍎 Smart Food Quality Detector")
st.markdown("Real-time fruit/vegetable classification with quality, shelf life, and recipe tips.")
st.sidebar.info("✅ Ensure:\n- Good lighting\n- Object is centered\n- Hold steady near webcam")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        disp = img.copy()

        try:
            resized = cv2.resize(img, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_array = np.expand_dims(rgb / 255.0, axis=0)

            fv_pred = fruit_veg_model.predict(input_array)
            fv_class = np.argmax(fv_pred)
            fv_label = "fruit" if fv_class == 0 else "vegetable"

            if fv_label == "fruit":
                pred = fruit_model.predict(input_array)
                label = fruit_le.inverse_transform([np.argmax(pred)])[0]
            else:
                pred = veg_model.predict(input_array)
                label = veg_classes[np.argmax(pred)]

            quality = label.lower()
            confidence = np.max(pred) * 100
            shelf_days = shelf_life_map.get(quality, "N/A")

            object_name = ""
            for key in tips:
                if key in label.lower():
                    object_name = key
                    break
            if not object_name:
                object_name = "banana"

            nutri, recipe = tips.get(object_name, {}).get(quality, ("No tip", "No recipe"))

            cv2.rectangle(disp, (10, 10), (640, 130), (255, 255, 255), -1)
            cv2.putText(disp, f"{fv_label.upper()} - {label.upper()} ({confidence:.1f}%)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
            cv2.putText(disp, f"📆 Shelf Life: {shelf_days} day(s)", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
            cv2.putText(disp, f"🍽 Tip: {nutri}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 150, 100), 1)
            cv2.putText(disp, f"🍳 Try: {recipe}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1)

        except Exception as e:
            print("❌ Error during prediction:", e)
            cv2.putText(disp, "❌ Prediction Error", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(disp, format="bgr24")

webrtc_streamer(key="full-app", video_processor_factory=VideoProcessor)
