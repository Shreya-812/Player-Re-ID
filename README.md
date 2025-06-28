# Player Re-Identification using YOLOv8 + ByteTrack

This project implements a real-time player re-identification system using YOLOv8 for detection and ByteTrack for consistent ID assignment across frames. It addresses the core challenge of assigning the same ID to players even after they leave and re-enter the frame.

## 📂 Project Structure

- `track.mp4` — Output video with tracked player IDs
- `player_reid_bytetrack.ipynb` — Jupyter notebook with the full pipeline
- `README.md` — Setup and usage guide
- `report.pdf` — Detailed explanation of methodology and approach

## 🔧 Setup Instructions

### Requirements

- Python ≥ 3.8
- [YOLOv8 / Ultralytics](https://docs.ultralytics.com/)
- OpenCV

### Install dependencies

```bash
pip install ultralytics opencv-python
```
### Run the Tracker
from ultralytics import YOLO
model = YOLO("yolov8m.pt")
model.track(source="15sec_input_720p.mp4", tracker="bytetrack.yaml", conf=0.25, save=True)

The output video will be saved to runs/track/exp*/track.mp4.

### Output Preview
The system detects and tracks players throughout the video, maintaining consistent IDs across brief occlusions or exits from frame.

### Tools & Techniques
1.YOLOv8 (object detection)
2.ByteTrack (re-identification tracking)
3.Confidence threshold tuning for sports-specific motion
4.Tested on provided 15-second clip

### Future Enhancements
1.Use of visual embeddings (CLIP / ReIDNet) for cross-camera mapping
2.Player jersey color filtering to distinguish teammates
3.Facial recognition or pose-guided embeddings for finer identification
