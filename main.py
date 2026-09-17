# ============================================================
# ChronoVision AI - Main Application
# Smart Classroom Monitoring System
# ============================================================

import cv2
from ultralytics import YOLO
import time
import os

from modules.attendance.attendance import mark_attendance
from modules.face_recognition.recognizer import recognize_face

# ---------------- CONFIGURATION ----------------
SCREENSHOT_DIR = "screenshots"
MODEL_PATH = "yolov8n.pt"
WINDOW_NAME = "ChronoVision AI"

# Performance Settings
FRAME_RESIZE = (480, 360)

YOLO_SKIP_FRAMES = 4
FACE_SKIP_FRAMES = 10

ALERT_COOLDOWN = 10

USE_GPU = False

# Ensure screenshot folder exists
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Initialize camera
def open_camera():
    attempts = [
        (0, cv2.CAP_DSHOW),
        (0, cv2.CAP_MSMF),
        (0, cv2.CAP_ANY),
        (1, cv2.CAP_DSHOW),
        (1, cv2.CAP_MSMF),
        (1, cv2.CAP_ANY),
    ]

    for index, backend in attempts:
        cap = cv2.VideoCapture(index, backend)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if cap.isOpened():
            print(f"✅ Camera opened: index={index}, backend={backend}")
            return cap
        
        cap.release()

    # Final fallback without backend
    for index in [0, 1, 2]:
        cap = cv2.VideoCapture(index)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if cap.isOpened():
            print(f"✅ Camera opened: index={index}, backend=default")
            return cap
        cap.release()

    return None

cap = open_camera()
if cap is None:
    print("❌ Camera error: unable to open any video capture device. Check camera connection, index, or whether another app is using it.")
    raise SystemExit(1)

# Runtime variables
last_alert_time = 0
frame_count = 0
current_student = "Unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera error: frame read failed")
        break

    frame_count += 1
    frame = cv2.resize(frame, FRAME_RESIZE)

    # ---------------- YOLO DETECTION ----------------
    phone_detected = False

    if frame_count % YOLO_SKIP_FRAMES == 0:
        results = model(frame, verbose=False)

        for box in results[0].boxes:
            cls = int(box.cls)
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if label == "cell phone":
                phone_detected = True

    # ---------------- FACE RECOGNITION ----------------
    if frame_count % FACE_SKIP_FRAMES == 0:
        current_student = recognize_face(frame)

    # ---------------- ALERT SYSTEM ----------------
    status = "No Violation"

    if current_student != "Unknown" and phone_detected:
        status = "⚠ Phone Usage Detected!"

        if time.time() - last_alert_time > ALERT_COOLDOWN:
            print(f"⚠ Phone usage detected by {current_student}")

            # Create filename
            filename = f"{SCREENSHOT_DIR}/violation_{int(time.time())}.jpg"

            # Save screenshot
            cv2.imwrite(filename, frame)
            print("📸 Screenshot saved:", filename)

            # Save attendance
            mark_attendance(current_student, filename)
            print("✅ Attendance saved")

            last_alert_time = time.time()

    # ---------------- DISPLAY ----------------
    cv2.putText(frame, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.putText(frame, current_student, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow(WINDOW_NAME, frame)

    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()