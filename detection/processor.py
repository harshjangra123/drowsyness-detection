import cv2
import mediapipe as mp
import numpy as np

# Initialize FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# ---- Utility functions ----
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_EAR(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    A = euclidean_distance(p2, p6)
    B = euclidean_distance(p3, p5)
    C = euclidean_distance(p1, p4)
    return (A + B) / (2.0 * C)

def calculate_MAR(landmarks, mouth_indices):
    top_lip, bottom_lip, left_corner, right_corner = [landmarks[i] for i in mouth_indices]
    A = euclidean_distance(top_lip, bottom_lip)
    C = euclidean_distance(left_corner, right_corner)
    return A / C

def calculate_head_tilt(landmarks, left_eye_indices, right_eye_indices):
    left_eye_center = np.mean([landmarks[i] for i in left_eye_indices], axis=0)
    right_eye_center = np.mean([landmarks[i] for i in right_eye_indices], axis=0)
    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    return np.degrees(np.arctan2(dy, dx))


# ---- Landmark indices ----
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]

# ---- Thresholds ----
EYE_AR_THRESH = 0.16
MAR_THRESH = 0.6
HEAD_TILT_THRESH = 10


# ---- Main detection function ----
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Default values (in case no face detected)
    eye_state, yawn_state, tilt_state = "No Face Detected", "-", "-"
    ear, mar, head_tilt, drowsiness_score = 0, 0, 0, 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]

            # === Calculate metrics ===
            left_ear = calculate_EAR(landmarks, LEFT_EYE)
            right_ear = calculate_EAR(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0
            mar = calculate_MAR(landmarks, MOUTH)
            head_tilt = calculate_head_tilt(landmarks, LEFT_EYE, RIGHT_EYE)

            # === Eye state ===
            eye_state = "Eyes Closed" if ear < EYE_AR_THRESH else "Eyes Open"
            eye_color = (0, 0, 255) if ear < EYE_AR_THRESH else (0, 255, 0)

            # === Yawn state ===
            yawn_state = "Yawning" if mar > MAR_THRESH else "Not Yawning"
            yawn_color = (0, 0, 255) if mar > MAR_THRESH else (0, 255, 0)

            # === Head tilt ===
            tilt_state = "Head Tilted" if abs(head_tilt) > HEAD_TILT_THRESH else "Head Straight"
            tilt_color = (0, 0, 255) if abs(head_tilt) > HEAD_TILT_THRESH else (0, 255, 0)

            # === Drowsiness Score ===
            ear_factor = max(0, min(1, (EYE_AR_THRESH - ear) / EYE_AR_THRESH))
            mar_factor = max(0, min(1, (mar / MAR_THRESH)))
            tilt_factor = max(0, min(1, abs(head_tilt) / HEAD_TILT_THRESH))

            drowsiness_score = (0.5 * ear_factor + 0.3 * mar_factor + 0.2 * tilt_factor) * 100
            drowsiness_score = int((min(drowsiness_score, 100))*1.5)

            # === Drowsiness Status ===
            if drowsiness_score < 30:
                status = "Awake"
                color = (0, 255, 0)
            elif drowsiness_score < 60:
                status = "Slightly Drowsy"
                color = (0, 255, 255)
            else:
                status = "Drowsy"
                color = (0, 0, 255)

            # === Display everything ===
            cv2.putText(frame, f"{eye_state}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, eye_color, 3)
            cv2.putText(frame, f"{yawn_state}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, yawn_color, 3)
            cv2.putText(frame, f"{tilt_state}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, tilt_color, 3)
            cv2.putText(frame, f"EAR: {ear:.2f} | MAR: {mar:.2f} | Tilt: {head_tilt:.1f}°",
                        (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Drowsiness Score: {drowsiness_score}", (30, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 3)
            cv2.putText(frame, f"Status: {status}", (30, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

    return frame, eye_state, yawn_state, tilt_state, ear, mar, head_tilt, drowsiness_score
