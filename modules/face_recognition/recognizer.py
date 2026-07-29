import face_recognition
import os
import cv2

known_encodings = []
known_names = []

# Load dataset
dataset_path = "dataset/students"

for student_name in os.listdir(dataset_path):

    student_folder = os.path.join(dataset_path, student_name)

    # Skip non-folders
    if not os.path.isdir(student_folder):
        continue

    # Read all images inside student folder
    for image_name in os.listdir(student_folder):

        img_path = os.path.join(student_folder, image_name)

        try:
            image = face_recognition.load_image_file(img_path)

            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(student_name)

        except Exception as e:
            print(f"Error loading {img_path}: {e}")

def recognize_face(frame):

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations
    face_locations = face_recognition.face_locations(
        rgb_frame,
        model="hog"
    )

    # Encode faces
    face_encodings = face_recognition.face_encodings(
        rgb_frame,
        face_locations
    )

    name = "Unknown"

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        distances = face_recognition.face_distance(known_encodings, face_encoding)

        if len(distances) > 0:
            best_match_index = distances.argmin()

            if matches[best_match_index] and distances[best_match_index] < 0.5:
                name = known_names[best_match_index]

    return name