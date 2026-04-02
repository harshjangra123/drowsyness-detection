import streamlit as st
import cv2
from detector.processor import process_frame

st.set_page_config(page_title="Drowsiness Detection", layout="centered")
st.title("Drowsiness Detection")
st.markdown("Real-time detection of **Eye Closure**, **Yawning**, and **Head Tilt** with Drowsiness Score")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

score_placeholder = st.empty()

cap = None
if run:
    cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Cannot access webcam")
        break

    frame = cv2.flip(frame, 1)
    processed_frame, eye_state, yawn_state, tilt_state, ear, mar, head_tilt, drowsiness_score = process_frame(frame)

    FRAME_WINDOW.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
    score_placeholder.metric("Drowsiness Score", f"{drowsiness_score}")

else:
    if cap:
        cap.release()
