# 🚗 Vehicle Detection and Counting using YOLOv8

This project detects and counts vehicles in real-time using a custom-trained YOLOv8 object detection model. The solution is built for analyzing traffic CCTV video feeds and supports vehicle tracking and classification for five vehicle types: **Bus, Car, Motorcycle, Pickup, and Truck**.

---

## 📁 Project Structure
├── vehicle-detection.ipynb # Main notebook for training and inference
├── dataset/
│ ├── train/
│ ├── valid/
│ ├── test/
│ └── data.yaml # Dataset configuration file
├── runs/
│ └── detect/
│ └── train3/
│ └── weights/
│ └── best.pt # Trained model weights
└── output_with_counts.mp4 # Output video with detection boxes

## 🧠 Model Info

- **Framework**: [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- **Model Used**: YOLOv8n (`yolov8n.yaml`)
- **Training Epochs**: 300
- **Dataset Classes**: Bus, Car, Motorcycle, Pickup, Truck
- **Metrics**:
  - Precision/Recall: Good for most classes, lower for Motorcycle/Truck

---

## ⚙️ Installation

```bash
pip install ultralytics opencv-python
