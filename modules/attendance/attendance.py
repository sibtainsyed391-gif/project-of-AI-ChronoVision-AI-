import sqlite3
import datetime
import csv
import os

def mark_attendance(name, image_path):
    now = datetime.datetime.now()
    date = now.strftime('%Y-%m-%d')
    time = now.strftime('%H:%M:%S')

    # -------- CSV SAVE --------
    csv_file = "modules/attendance/attendance.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])
        writer.writerow([name, date, time])

    # -------- DATABASE SAVE --------
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            date TEXT,
            time TEXT,
            image TEXT
        )
    """)

    cursor.execute(
        "INSERT INTO attendance (name, date, time, image) VALUES (?, ?, ?, ?)",
        (name, date, time, image_path)
    )

    conn.commit()
    conn.close()