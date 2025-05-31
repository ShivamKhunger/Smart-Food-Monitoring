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

veg_classes = ['fresh', 'stale']
fruit_veg_classes = ['apple', 'banana', 'orange', 'broccoli', 'carrot']

shelf_life_map = {
    'fresh': 5, 'ripe': 3, 'overripe': 1, 'unripe': 6, 'stale': 0
}

tips = {
    "banana": {
        "ripe": ("Energy booster 🍌", "Use in smoothie"),
        "overripe": ("Good for digestion", "Make banana bread 🍞"),
        "stale": ("Avoid eating", "Compost it")
    },
    "apple": {
        "fresh": ("Rich in fiber", "Eat raw or in salad 🥗"),
        "stale": ("Low nutrition", "Make jam or cider")
    },
    "orange": {
        "fresh": ("Vitamin C rich 🍊", "Juice or raw"),
        "stale": ("Unsafe to eat", "Use peel for cleaning")
    }
}

yolo = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("📷 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    results = yolo.predict(frame, conf=0.25, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = yolo.names[cls_id].lower()

        if label in fruit_veg_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            if crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            resized = cv2.resize(crop, (224, 224))
            input_array = np.expand_dims(resized / 255.0, axis=0)

            fv_pred = fruit_veg_model.predict(input_array)
            fv_label = "fruit" if np.argmax(fv_pred) == 0 else "vegetable"

            if fv_label == "fruit":
                pred = fruit_model.predict(input_array)
                quality = fruit_le.inverse_transform([np.argmax(pred)])[0]
            else:
                pred = veg_model.predict(input_array)
                quality = veg_classes[np.argmax(pred)]

            confidence = np.max(pred) * 100
            shelf = shelf_life_map.get(quality.lower(), "N/A")
            tip, recipe = tips.get(label, {}).get(quality.lower(), ("", ""))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label.upper()} | {fv_label.upper()} | {quality.upper()} ({confidence:.1f}%)",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 255, 10), 2)
            cv2.putText(frame, f"Shelf Life: {shelf} days", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
            if tip:
                cv2.putText(frame, f"Tip: {tip}", (x1, y2 + 40),
                            cv2.FONT_HERSHEY_PLAIN, 1, (200, 255, 100), 1)
            if recipe:
                cv2.putText(frame, f"Try: {recipe}", (x1, y2 + 60),
                            cv2.FONT_HERSHEY_PLAIN, 1, (150, 200, 255), 1)

    cv2.imshow("YOLOv8 Food Quality Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
