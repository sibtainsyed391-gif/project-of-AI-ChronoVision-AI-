# ChronoVision AI main system controller

import cv2

from modules.phone_detection.phone_detection import detect_phone
from modules.face_recognition.face_detection import detect_face
from modules.attendance.attendance import mark_attendance


def start_system():
    print("ChronoVision AI system starting...")


def load_models():
    print("Loading models...")


def run_detection():
    print("Running detection modules...")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        phone, person, frame = detect_phone(frame)
        student = detect_face(frame)

        if phone and person:
            print(f"⚠ Phone detected by {student}")
            mark_attendance(student)

        cv2.imshow("ChronoVision AI", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def save_results():
    print("Saving results...")


def shutdown():
    print("System shutting down...")


def main():
    start_system()
    load_models()
    run_detection()
    save_results()
    shutdown()


if __name__ == "__main__":
    main()