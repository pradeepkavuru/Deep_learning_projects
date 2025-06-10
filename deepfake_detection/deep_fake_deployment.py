import cv2
import streamlit as st
import tempfile
import numpy as np
from tensorflow.keras.models import load_model


model = load_model(r"deepfake_detection/new_fake.h5")  
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    

def predict_face(face):
    face = cv2.resize(face, (256, 256))
    face = np.reshape(face, (1, 256, 256, 3))
    face = face / 255.0
    return model.predict(face)[0][0]


st.title("🎭 Deepfake Detection")
video_file = st.file_uploader("📂 Upload a Video", type=["mp4", "avi", "mov"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    predictions = []
    face_frame_count = 0
    

    while cap.isOpened() and face_frame_count < 30:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = frame[y:y+h, x:x+w]
            if face.size > 0:
                score = predict_face(face)
                predictions.append(score)
                color = (255, 255, 255) 
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                stframe.image(frame, channels="BGR", caption=f"Face Frame {face_frame_count+1}/30")
                face_frame_count += 1

    

    cap.release()

    if predictions:
        mean_score = np.mean(predictions)
        label = 'Real' if mean_score > 0.5 else 'Fake'
        confidence = mean_score if mean_score > 0.5 else 1 - mean_score
        st.markdown("---")
        st.subheader(f"🧠 Final Prediction: **{label}** ({confidence * 100:.2f}%)")
    else:
        st.warning("No faces detected in the video.")
