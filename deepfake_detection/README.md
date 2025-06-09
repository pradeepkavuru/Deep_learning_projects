# 🎭 Deepfake Detection from Video using CNN + Streamlit

This project is a deep learning-based application that detects **deepfake videos** by analyzing facial features frame-by-frame. It uses OpenCV and Haar cascades for face detection, a CNN-based model (or transfer learning like VGG/ResNet) for prediction, and a **Streamlit** interface for real-time interaction.

---

## 📌 Features

- 🎥 Accepts a video file (mp4, avi, etc.)
- 🧠 Detects faces from frames using Haar Cascade
- 📊 Predicts whether each face is **Real** or **Fake**
- 🖼️ Shows bounding boxes on face detections
- ✅ Final prediction shown after analyzing 30 face frames
- 🌐 Built with Streamlit for a simple UI

---

## 🧠 Model Details

- Input: Detected face images (256×256×3)
- Architecture: CNN (or pretrained ResNet/VGG + Dense layers)
- Output: Binary classification - **Real** (label 1) or **Fake** (label 0)
- Activation: Sigmoid
- Loss: Binary Crossentropy
- Trained on: Real vs Deepfake face datasets
- link: https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images

---

## 🛠️ Tech Stack

- Python
- OpenCV (for video/face processing)
- TensorFlow / Keras (model)
- Streamlit (web UI)
- NumPy, os, tempfile, cv2

---

## 🖼️ How It Works

1. User uploads a **video**
2. Frames are extracted
3. Faces are detected using **Haar Cascades**
4. First **30 face frames** are selected
5. Each face is **preprocessed and fed into the model**
6. Bounding boxes are drawn on each frame (no per-frame label shown)
7. The average prediction score is used to output a **final label**:
   - **Real** if mean prediction > 0.5
   - **Fake** otherwise

---
