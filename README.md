# Gymmy - Robotic Physical Therapy Assistant

A robotic system for physical therapy rehabilitation exercises, featuring personalized Range of Motion (ROM) assessment.

## Overview

Gymmy is an interactive robotic assistant designed to help patients perform physical therapy exercises. The system uses computer vision to track patient movements and provides real-time feedback through a humanoid robot.

### Key Features

- **Real-time Movement Tracking** - Uses ZED camera for skeleton detection
- **Robotic Demonstration** - Humanoid robot demonstrates exercises
- **Personalized ROM Assessment** - Measures each patient's range of motion
- **Adaptive Thresholds** - Adjusts exercise parameters based on patient capabilities
- **Progress Tracking** - Records and exports training data
- **Multi-exercise Support** - Ball, stick, rubber band, and weight exercises

## System Requirements

### Hardware
- ZED Stereo Camera
- Robotis OP2 (or compatible humanoid robot)
- Windows PC with NVIDIA GPU (CUDA support)

### Software
- Python 3.8+
- CUDA Toolkit
- ZED SDK

## Installation

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/gymmy.git
cd gymmy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install ZED SDK from [Stereolabs](https://www.stereolabs.com/developers/release/)

4. Run the application:
```bash
python main.py
```

## Project Structure

```
gymmy/
├── main.py                 # Application entry point
├── Camera.py               # ZED camera and skeleton tracking
├── ScreenNew.py            # Tkinter GUI
├── TrainingNew.py          # Training session logic
├── Gymmy.py                # Robot control
├── Excel.py                # Data storage and export
├── ExerciseConfig.py       # Exercise configuration (NEW)
├── CalibrationLogic.py     # ROM calculation algorithms (NEW)
├── PilotDataExport.py      # Pilot study data export (NEW)
├── Settings.py             # Global settings
├── Patients/               # Patient data folders
├── Pictures/               # UI images
└── audio files/            # Audio feedback files
```

## ROM Assessment Module

The ROM (Range of Motion) Assessment Module is a key addition that enables personalized exercise thresholds.

### How It Works

1. **Assessment Phase**: Patient performs exercises freely (no feedback)
2. **Data Collection**: System records angle measurements
3. **Threshold Calculation**: Algorithm detects peaks (UP) and valleys (DOWN)
4. **Personalization**: Calculated thresholds are saved for future sessions

### Algorithm Overview

```
1. Find local peaks (UP position) in angle data
2. Find local valleys (DOWN position)
3. Filter outliers (15% threshold)
4. Calculate: threshold = average ± 10°
5. Save to Patients.xlsx
```

### Files Added for ROM Module

| File | Description |
|------|-------------|
| `ExerciseConfig.py` | Maps exercises to angles and default thresholds |
| `CalibrationLogic.py` | Peak/valley detection and threshold calculation |
| `PilotDataExport.py` | Export pilot study data to Excel |

## Usage

### For Therapists

1. Register new patient (enter patient ID)
2. Select exercises for the patient
3. Initiate ROM Assessment (first session)
4. Start regular training sessions

### For Patients

1. Enter patient ID
2. Follow robot demonstrations
3. Perform exercises while system tracks movement
4. Receive real-time feedback

## Data Export

### Patient Data Location
```
Patients/[PatientID]/
├── ROM_Assessments/[date]/    # ROM assessment data
├── Trainings/[date]/          # Regular training data
└── PDF_to_Therapist_Email/    # Generated reports
```

### Pilot Study Export
```bash
python PilotDataExport.py           # Export data
python PilotDataExport.py analyze   # Run analysis
python PilotDataExport.py protocol  # View protocol
```

## Configuration

### Exercise Configuration

Exercises are configured in `ExerciseConfig.py`:

```python
EXERCISE_CONFIG = {
    "ball_bend_elbows": {
        "category": "ball",
        "angles": [
            {"index": 0, "side": "right", "default_up": 140, "default_down": 40},
            {"index": 1, "side": "left", "default_up": 140, "default_down": 40},
        ],
    },
    # ... more exercises
}
```

## Contributing

This project was developed as a final project for Ben-Gurion University.

## Acknowledgments

- Stereolabs for ZED SDK
- Robotis for OP2 robot platform
- Prof. Yael Edan and Dr. Shirley Handelzalts for project guidance


