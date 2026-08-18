# AI Yoga Pose Analyzer

A real-time computer vision and machine learning system that detects yoga poses from a webcam, classifies them using an ML model, and provides basic pose-form feedback.

The system combines **MediaPipe Pose Landmarks**, **feature engineering**, and **Random Forest classification** to recognize:

- Plank
- Tree
- Warrior II
- Unknown / Non-target pose

It also calculates a basic **Form Score (0–100)** and provides real-time corrective feedback.

---

##  Features

- Real-time webcam pose detection
- 33-point human pose landmark detection using MediaPipe
- Custom pose feature engineering
- Machine learning-based pose classification
- 4-class classification:
  - Plank
  - Tree
  - Warrior II
  - Unknown
- Prediction confidence
- Temporal prediction smoothing
- Pose-specific form analysis
- Form score from 0–100
- Real-time corrective feedback
- Modular Python architecture

---

## System Architecture

```text
                    Webcam
                       │
                       ▼
              ┌─────────────────┐
              │    MediaPipe    │
              │  Pose Landmarks │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Feature         │
              │ Engineering     │
              │                 │
              │ 18 Features     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Random Forest   │
              │ Classifier      │
              └────────┬────────┘
                       │
              ┌────────┴─────────┐
              │                  │
              ▼                  ▼
        Pose Prediction      Confidence
              │
              ▼
      ┌──────────────────┐
      │ Form Analyzer    │
      └────────┬─────────┘
               │
               ▼
        Form Score /100
               │
               ▼
       Corrective Feedback
       