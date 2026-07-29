ChronoVision AI
Smart Classroom Monitoring System Using AI

ChronoVision AI is an AI-powered smart classroom monitoring system designed to detect mobile phone usage, recognize students using face recognition, automatically log attendance violations, and display analytics through a modern Flask web dashboard.

The project combines:

YOLOv8 Object Detection
Face Recognition
SQLite Database
Flask Web Dashboard
Real-Time Monitoring
Screenshot Evidence System
🚀 Features
✅ Real-Time Phone Detection
Detects mobile phones using YOLOv8
Detects persons in classroom
Works in real-time using webcam
✅ Face Recognition
Recognizes registered students
Supports multiple images per student
Uses facial encodings for matching
Unknown faces remain unrecognized
✅ Automatic Attendance Logging
Saves:
Student Name
Date
Time
Screenshot Path
Stores records inside SQLite database
✅ Screenshot Evidence
Automatically captures screenshot when violation occurs
Screenshots stored in screenshots/
✅ Flask Dashboard
Modern UI dashboard
Displays violation records
Displays screenshots
Auto-refresh system
Charts and analytics support
✅ Performance Optimizations
Frame skipping
Reduced resolution
HOG-based recognition for CPU systems
Optimized real-time processing
🧠 Technologies Used
Technology	Purpose
Python	Core programming language
OpenCV	Video processing
YOLOv8	Object detection
face_recognition	Face recognition
Flask	Web framework
SQLite	Database
HTML/CSS	Frontend UI
Chart.js	Dashboard charts
📂 Project Structure
ChronoVision_AI/
│
├── main.py
├── app.py
├── attendance.db
├── yolov8n.pt
│
├── dataset/
│   └── students/
│       ├── sibtain_56/
│       │   ├── 1.jpg
│       │   ├── 2.jpg
│       │   └── ...
│       │
│       └── ali_12/
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
⚙️ System Workflow
Camera captures classroom frames
YOLOv8 detects:
Person
Cell Phone
Face Recognition identifies student
If phone usage detected:
Screenshot saved
Attendance violation logged
Database updated
Flask dashboard displays records
📸 Dataset Structure

Each student must have a separate folder:

students/
├── sibtain_56/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
│
├── ali_12/
│   ├── 1.jpg
│   └── ...
Dataset Recommendations
20–50 images per student minimum
Include:
Front face
Side angles
Different lighting
Classroom distance
Slight head movement
🖥️ Installation
1️⃣ Clone Project
git clone <repository-url>
cd ChronoVision_AI
2️⃣ Install Dependencies
pip install -r requirements.txt

Or manually:

pip install opencv-python
pip install ultralytics
pip install face_recognition
pip install flask
pip install flask-cors
pip install numpy
pip install pi-heif
▶️ Running the Project
Start Complete System
python app.py

Flask Dashboard:

http://127.0.0.1:5000

The detection engine automatically starts in the background.

🧩 Main Components
main.py

Handles:

Camera processing
YOLO detection
Face recognition
Screenshot capture
Attendance logging
app.py

Handles:

Flask server
Dashboard UI
Database fetching
Screenshot display
attendance.py

Handles:

SQLite database insertion
Attendance records
recognizer.py

Handles:

Dataset loading
Face encoding
Student recognition
📊 Database Structure
attendance Table
Column	Description
id	Primary Key
name	Student Name
date	Violation Date
time	Violation Time
image	Screenshot Path
⚡ Optimizations Used
Frame Skipping

Processes selected frames instead of every frame.

Reduced Resolution

Frames resized to improve FPS.

HOG Face Detection

Used instead of CNN because:

Faster on CPU
Lower resource usage
Works without GPU
🔥 Future Improvements
GPU acceleration
Cloud deployment
Mobile application integration
Email/SMS alerts
Live analytics dashboard
Emotion detection
Multi-camera support
Custom-trained YOLO model
🎓 Academic Objective

The purpose of ChronoVision AI is to:

Improve classroom discipline
Automate attendance systems
Monitor mobile usage in classrooms
Demonstrate practical AI implementation using computer vision
👥 Team Contributions
Member	Responsibility
Member 1	Face Recognition Module
Member 2	YOLOv8 Phone Detection
Member 3	Attendance & Database
Member 4	Flask Dashboard & UI
Member 5	System Integration & Optimization
🧠 AI Concepts Used
Computer Vision
Object Detection
Face Recognition
Facial Encodings
Real-Time Processing
Deep Learning
Machine Learning
📌 Important Notes
Good lighting improves accuracy
Multiple images improve recognition performance
CPU systems may experience lag without optimization
GPU support can significantly improve performance
📜 License

This project was developed for educational and academic purposes.

👨‍💻 Project Name
ChronoVision AI
Smart Classroom Monitoring System