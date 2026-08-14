# ChronoVision AI

### Smart Classroom Monitoring System Using Artificial Intelligence

🏆 **1st Position — Mini-Project Competition**
**University of Sindh, MBBS Campus Dadu**

ChronoVision AI is an AI-powered classroom monitoring system designed to automate classroom monitoring using computer vision and artificial intelligence.

The system combines **YOLOv8 object detection, face recognition, real-time video processing, SQLite, and a Flask web dashboard** to detect mobile phone usage, identify registered students, record violation events, and provide screenshot evidence through a centralized dashboard.

---

## 🚀 What Does ChronoVision AI Do?

ChronoVision AI monitors a classroom camera feed in real time and performs several tasks automatically:

* 📱 Detects mobile phone usage using YOLOv8
* 👤 Detects people in the classroom
* 🧑‍💻 Recognizes registered students using face recognition
* 📝 Records violation events automatically
* 📸 Captures screenshot evidence when a violation is detected
* 🗄️ Stores records in a SQLite database
* 📊 Displays records and screenshots through a Flask dashboard
* ⚡ Uses performance optimizations to improve CPU-based operation

---

## 🎯 Problem We Wanted to Solve

Traditional classroom monitoring can be time-consuming and difficult to perform continuously.

Teachers may have to divide their attention between teaching and monitoring student behavior. Manual attendance and violation recording can also take additional time.

ChronoVision AI explores how computer vision and AI can automate parts of this process by continuously analyzing a classroom camera feed and recording relevant events.

---

## 🧠 How the System Works

```text
                 Camera / Webcam
                       │
                       ▼
              Real-Time Video Frames
                       │
                       ▼
                  YOLOv8 Detection
                  ┌────────┴────────┐
                  ▼                 ▼
               Person          Mobile Phone
                  │                 │
                  ▼                 ▼
            Face Recognition    Violation Event
                  │                 │
                  ▼                 ▼
            Student Identity    Screenshot
                  │                 │
                  └────────┬────────┘
                           ▼
                    SQLite Database
                           │
                           ▼
                    Flask Dashboard
                           │
                           ▼
                Records + Analytics
```

---

## ✨ Key Features

### 📱 Real-Time Phone Detection

YOLOv8 is used to detect objects in the classroom camera feed, including mobile phones and people.

The system processes the video stream continuously and identifies relevant objects in real time.

### 👤 Face Recognition

The system recognizes registered students using facial encodings.

Multiple images can be used for each registered student to improve recognition under different conditions.

Unknown faces remain unrecognized.

### 📝 Automatic Violation Logging

When a relevant phone-usage event is detected, the system records information such as:

* Student name
* Date
* Time
* Screenshot path

The records are stored in a SQLite database.

### 📸 Screenshot Evidence

When a violation event is detected, the system automatically captures a screenshot.

These screenshots provide visual evidence associated with the recorded event.

### 🖥️ Flask Web Dashboard

A Flask-based dashboard provides a centralized interface for viewing:

* Violation records
* Student information
* Screenshot evidence
* Dashboard analytics

### ⚡ Performance Optimization

The system includes several optimizations to make real-time processing more practical on CPU-based systems:

* Frame skipping
* Reduced video resolution
* HOG-based face detection
* Optimized real-time processing

---

## 🛠️ Technology Stack

| Technology       | Purpose                     |
| ---------------- | --------------------------- |
| Python           | Core programming language   |
| OpenCV           | Video and camera processing |
| YOLOv8           | Object detection            |
| face_recognition | Face recognition            |
| Flask            | Web dashboard/backend       |
| SQLite           | Database                    |
| HTML/CSS         | Frontend                    |
| Chart.js         | Dashboard charts            |

---

## 📸 System Screenshots

### Real-Time Detection

*Add your phone/person detection screenshot here.*

```text
[Detection Screenshot]
```

### Face Recognition

*Add your face recognition screenshot here.*

```text
[Face Recognition Screenshot]
```

### Violation Evidence

*Add a screenshot showing the captured violation evidence here.*

```text
[Violation Screenshot]
```

### Flask Dashboard

*Add your dashboard screenshot here.*

```text
[Dashboard Screenshot]
```

---

## 📂 Project Structure

```text
ChronoVision_AI/
│
├── main.py
├── app.py
├── attendance.db
├── yolov8n.pt
│
├── dataset/
│   └── students/
│       ├── student_1/
│       │   ├── 1.jpg
│       │   ├── 2.jpg
│       │   └── ...
│       │
│       └── student_2/
│           ├── 1.jpg
│           └── ...
│
├── modules/
│   ├── attendance/
│   │   └── attendance.py
│   │
│   ├── face_recognition/
│   │   ├── recognizer.py
│   │   └── Trainer.py
│   │
│   └── phone_detection/
│       └── phone_detection.py
│
├── screenshots/
│
├── static/
│   └── style.css
│
└── templates/
    └── index.html
```

---

## ⚙️ System Workflow

1. The camera captures classroom frames.
2. YOLOv8 analyzes the frames and detects people and mobile phones.
3. Face recognition attempts to identify registered students.
4. When a relevant phone-usage event is detected, the system captures a screenshot.
5. The event is recorded in the SQLite database.
6. The Flask dashboard retrieves and displays the recorded information.

---

## 👨‍💻 My Contribution

ChronoVision AI was originally developed as a **5-person team project**, with different members responsible for different modules.

The project was divided into areas including:

* Face recognition
* YOLOv8 phone detection
* Attendance/database
* Flask dashboard
* System integration

My main contribution was **system integration and debugging**.

The individually developed modules initially had compatibility and integration issues. I took responsibility for bringing the separate components together into one functioning end-to-end system.

I worked on:

* Integrating the different AI modules
* Debugging compatibility issues
* Connecting detection with face recognition
* Connecting events with the database
* Connecting the backend with the Flask dashboard
* Improving the overall project structure
* Optimizing the system for practical CPU-based execution
* Testing the complete workflow

This experience gave me practical experience in taking separate software components and turning them into a working AI application.

---

## 🏆 Achievement

### 1st Position — Mini-Project Competition

**University of Sindh, MBBS Campus Dadu**

ChronoVision AI achieved **1st position** in our mini-project competition.

---

## 🧩 Main Components

### `main.py`

Responsible for the main monitoring and processing workflow, including:

* Camera processing
* YOLO detection
* Face recognition
* Screenshot capture
* Attendance/violation logging

### `app.py`

Responsible for:

* Flask server
* Dashboard interface
* Database retrieval
* Screenshot display

### `attendance.py`

Responsible for:

* SQLite database operations
* Attendance/violation records

### `recognizer.py`

Responsible for:

* Loading the student dataset
* Generating face encodings
* Student recognition

---

## 📊 Database

The project uses SQLite to store monitoring records.

### Attendance Table

| Column  | Description     |
| ------- | --------------- |
| `id`    | Primary key     |
| `name`  | Student name    |
| `date`  | Violation date  |
| `time`  | Violation time  |
| `image` | Screenshot path |

---

## 📸 Student Dataset

Each registered student has a separate directory containing their images.

Example:

```text
students/
├── student_1/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
│
└── student_2/
    ├── 1.jpg
    └── ...
```

Multiple images can be used to represent different:

* Face angles
* Lighting conditions
* Distances
* Head positions

---

## 🖥️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ChronoVision_AI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install the main dependencies manually:

```bash
pip install opencv-python
pip install ultralytics
pip install face_recognition
pip install flask
pip install flask-cors
pip install numpy
pip install pi-heif
```

---

## ▶️ Running the Project

Start the system with:

```bash
python app.py
```

The Flask dashboard will be available at:

```text
http://127.0.0.1:5000
```

The detection engine automatically starts in the background.

---

## ⚡ Performance Optimizations

### Frame Skipping

Instead of processing every frame, selected frames are processed to reduce computational requirements.

### Reduced Resolution

Video frames can be resized before processing to improve performance.

### HOG Face Detection

HOG-based face detection was used because it is more suitable for CPU-based execution than heavier CNN-based approaches.

These optimizations help the system operate more efficiently on systems without dedicated GPU hardware.

---

## 🔮 Future Improvements

Potential future improvements include:

* GPU acceleration
* Cloud deployment
* Mobile application integration
* Email/SMS alerts
* Advanced live analytics
* Emotion detection
* Multi-camera support
* Custom-trained YOLO model

---

## 🎓 Academic Objective

ChronoVision AI was developed to demonstrate practical applications of artificial intelligence and computer vision.

The project focuses on:

* Classroom monitoring
* Automated attendance/violation recording
* Mobile phone detection
* Face recognition
* Real-time computer vision
* AI system integration

---

## 🧠 AI Concepts Demonstrated

* Computer Vision
* Object Detection
* Face Recognition
* Facial Encodings
* Deep Learning
* Machine Learning
* Real-Time Video Processing
* AI System Integration

---

## ⚠️ Important Notes

System performance and recognition accuracy can depend on:

* Lighting conditions
* Camera quality
* Face image quality
* Number of registered images
* Processing hardware

GPU acceleration can significantly improve performance for larger deployments.

---

## 📜 License

This project was developed for educational and academic purposes.

---

## 👨‍💻 Project

**ChronoVision AI**
Smart Classroom Monitoring System Using Artificial Intelligence

🏆 **1st Position — University of Sindh, MBBS Campus Dadu**
