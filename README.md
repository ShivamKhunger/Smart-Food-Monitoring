🥗 Smart Food Quality Detection Using Deep Learning & Computer Vision
A real-time, consumer-ready AI system that detects food items and classifies their quality stage (e.g., fresh, ripe, overripe, stale) using computer vision and deep learning. It also predicts estimated shelf life and provides nutritional tips and recipe suggestions — all from a live webcam feed.

🚀 Project Summary
Manual food inspection is slow, subjective, and not scalable. Existing automated systems are often expensive, binary, or limited to industrial settings. This project proposes a lightweight, extensible, and real-time solution using open-source technologies.

✅ Key Features
Real-Time Detection: Webcam-based object detection using YOLOv8.

Multi-Stage Classification:

Fruits: Fresh, Ripe, Unripe, Overripe, Stale

Vegetables: Fresh, Stale

Dual Classification Pipeline: First identify whether it's a fruit or vegetable, then classify its quality.

Shelf Life Estimation: Predict remaining days using predefined quality-stage mappings.

Nutritional Tip Generator: Hardcoded health advice based on item and quality.

Recipe Suggestions: Dictionary-based recommendation engine using food type and stage.

🧠 Tech Stack
YOLOv8 for object detection (via Ultralytics)

MobileNetV2 for classification (with custom dense layers)

Streamlit for GUI and webcam interface

TensorFlow/Keras for model training

Python, OpenCV, NumPy, Matplotlib

📊 Dataset
Size: 50,000+ labeled images

Sources: Kaggle + custom-collected images

Categories: Banana, Mango, Cucumber, Carrot, Apple, Broccoli (and more)

Preprocessing:

Resized to 224x224

RGB normalization

Data augmentation: Rotation, flipping, brightness

📈 Model Performance
Task	Accuracy
Fruit vs Vegetable	99.1%
Vegetable Quality (2-class)	97.4%
Fruit Quality (5-class)	92.8%

🛠️ Current Status
✅ Data preprocessing completed

✅ All models trained and tested

✅ Streamlit-based real-time system working

✅ Detection + classification + suggestion modules integrated

✅ Output tested on multiple food types

🧪 Model evaluation using accuracy (future: precision, recall, F1)

📦 Final polishing and packaging in progress

🔮 Upcoming Features
🔁 Model performance improvement via more data and fine-tuning

🌾 Extension to grains, dairy, and packaged foods

🌙 Better performance in low-light conditions

📱 Deployment on Raspberry Pi or Android (lightweight version)

📊 Class-wise precision & recall reporting
& many more.
