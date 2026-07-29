import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    results = model(frame)

    phone_detected = False
    person_detected = False

    for box in results[0].boxes:
        cls = int(box.cls)
        label = model.names[cls]

        if label == "cell phone":
            phone_detected = True

        if label == "person":
            person_detected = True

    # 🚨 ALERT LOGIC
    if phone_detected and person_detected:
        print("⚠ Phone usage detected!")

    annotated = results[0].plot()
    cv2.imshow("ChronoVision AI", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()