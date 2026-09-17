---
title: ChronoVision AI
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: python
app_file: app.py
pinned: false
python_version: "3.10"
---

# ChronoVision AI

A smart AI-based attendance and monitoring system for classroom environments.

This project combines computer vision, face recognition, and phone detection to help automate attendance tracking and classroom monitoring.

## Features
- Real-time student attendance tracking
- Face recognition-based identification
- Phone detection in class
- Attendance analytics dashboard
- Local SQLite database support

## Run locally
```bash
python app.py
```

## Project structure
- `app.py` — main Flask web app
- `main.py` — detection pipeline
- `modules/` — AI detection and attendance logic
- `static/` and `templates/` — frontend files
