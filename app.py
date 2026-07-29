import sqlite3
from collections import Counter
from flask import Flask, render_template, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DB_NAME = 'attendance.db'

# ---------------- DATABASE INIT ----------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            date TEXT,
            time TEXT,
            image TEXT
        )
    ''')

    conn.commit()
    conn.close()


# ---------------- SCREENSHOTS ----------------
@app.route('/screenshots/<path:filename>')
def screenshots(filename):
    return send_from_directory('screenshots', filename)

# ---------------- HOME PAGE ----------------
def fetch_attendance():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM attendance ORDER BY id DESC")
    records = cursor.fetchall()

    conn.close()
    return records


@app.route('/')
def index():
    records = fetch_attendance()
    return render_template('index.html', attendance=records)


@app.route('/analytics')
def analytics():
    records = fetch_attendance()

    names = [record[1] for record in records]
    student_counts = dict(Counter(names))

    dates = [record[2] for record in records]
    date_counts = dict(Counter(dates))

    top_student = max(student_counts, key=student_counts.get) if student_counts else "N/A"

    return render_template(
        'analytics.html',
        attendance=records,
        student_labels=list(student_counts.keys()),
        student_values=list(student_counts.values()),
        date_labels=list(date_counts.keys()),
        date_values=list(date_counts.values()),
        top_student=top_student,
        top_student_count=student_counts.get(top_student, 0)
    )


# ---------------- RUN APP ----------------
if __name__ == '__main__':
    init_db()
    
    # Start main.py detection in background thread
    import threading
    import subprocess
    
    def run_detection():
        subprocess.run(["python", "main.py"])
    
    detection_thread = threading.Thread(target=run_detection, daemon=True)
    detection_thread.start()
    print("🔍 Detection started in background...")
    
    # Run Flask app
    app.run(debug=True, port=5000)


