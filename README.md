# Multi-Object Detection and Tracking System (YOLOv8 + SORT)

An industrial-grade, real-time object detection and tracking pipeline built using **Python, OpenCV, Ultralytics YOLOv8**, and a custom-implemented **SORT (Simple Online and Realtime Tracking)** algorithm.

This project is prepared and structured for internship submissions, featuring a modular code layout, comprehensive documentation of algorithms, and dynamic visual overlays.

---

## 🌟 Key Features
- **Real-Time Bounding Box Association**: Tracks objects frame-by-frame, assigning a unique tracking ID that remains consistent even under partial occlusions.
- **State-of-the-Art Object Detection**: Utilizes **YOLOv8** (pre-trained on the COCO dataset) for high-accuracy class detections (e.g., Person, Car, Bus, Motorcycle, Truck).
- **Custom-Built Kalman Filter**: An implementation of Kalman filtering written using pure NumPy to predict object movement, removing external libraries like `filterpy` for maximum dependency compatibility.
- **Optimal Association Solver**: Associates frame detections with tracker predictions using the Hungarian Algorithm (`scipy.optimize.linear_sum_assignment`) and an Intersection over Union (IoU) cost matrix.
- **Dynamic Visual Dashboard**: Displays active object counters per category, current execution frames per second (FPS), and total counts in a translucent overlay banner.
- **Auto-Save Output**: Automatically processes inputs and compiles tracked outputs to the `output_videos/` directory as H.264/MP4 files.

---

## 🛠️ Technology Stack
* **Language**: Python 3.10+
* **Deep Learning Framework**: PyTorch & Ultralytics YOLOv8
* **Computer Vision**: OpenCV (Open Source Computer Vision Library)
* **Mathematical Modeling**: NumPy (Linear algebra and state vector transformations)
* **Optimization Algorithms**: SciPy (Hungarian matching optimizer)

---

## 📁 Project Structure

```
Object_Detection_Tracking/
│
├── models/              # Cached YOLOv8 weights (e.g., yolov8n.pt)
├── input_videos/        # Raw input video files for tracking
├── output_videos/       # Saved tracked video outputs
├── screenshots/         # Screenshots of execution for reports
│
├── main.py              # Main orchestration and frame processing pipeline
├── tracker.py           # Custom SORT tracker with Kalman Filter & IoU association
├── requirements.txt     # Python package requirements
└── README.md            # Project documentation and guide
```

---

## 🧠 System Architecture & Algorithm Details

### 1. Object Detection (YOLOv8)
Object detection is performed by `ultralytics` YOLOv8. The model parses each frame, extracting bounding boxes coordinates $(x_1, y_1, x_2, y_2)$, confidence scores ($c$), and class classifications ($ID$). We filter detections to target specific classes of interest:
* **Vehicles**: Cars, Trucks, Buses, Motorcycles, Bicycles
* **Pedestrians**: Persons

### 2. Motion Prediction (Kalman Filtering)
To track objects when they move or suffer temporary occlusions, the state of each object is modeled using a linear Kalman filter with a constant velocity model.
* **State Vector ($x$)**: Represents the state of the object in 7 dimensions:
  $$x = [u, v, s, r, \dot{u}, \dot{v}, \dot{s}]^T$$
  Where $(u, v)$ is the bounding box center, $s$ is the scale (area), $r$ is the aspect ratio (width/height), and $\dot{u}, \dot{v}, \dot{s}$ are the respective velocities.
* **Measurement Vector ($z$)**: Directly observed bounding box properties:
  $$z = [u, v, s, r]^T$$
* **Prediction Phase**:
  $$x = Fx$$
  $$P = FPF^T + Q$$
* **Correction/Update Phase**:
  $$y = z - Hx$$
  $$S = HPH^T + R$$
  $$K = PS^T S^{-1}$$
  $$x = x + Ky$$
  $$P = (I - KH)P$$

### 3. Data Association (Hungarian Algorithm)
Detections in the current frame must be matched with active tracks from previous frames:
1. An **Intersection over Union (IoU) Matrix** is built between all predicted states and current detections.
2. The assignment is cast as a bipartite matching problem to maximize IoU, solved optimally using the Hungarian Algorithm (`linear_sum_assignment`).
3. Matches with an IoU score lower than `iou_threshold` (default: 0.3) are rejected. Unmatched detections are spawned as new tracks, and tracks that do not receive updates for `max_age` frames are terminated.

---

## 🚀 Installation Guide

### Prerequisites
Make sure you have Python installed. It is recommended to use a virtual environment.

### Step 1: Clone or Set Up Workspace
Ensure your files are placed in the appropriate structure:
```bash
git clone https://github.com/abakaushik-lgtm/ObjectDetectionAndTracking.git
cd ObjectDetectionAndTracking
```

### Step 2: Install Dependencies
Install all required libraries using the package manager `pip`:
```bash
pip install -r requirements.txt
```

---

## 💻 How to Run the Project

### 1. Tracking on Webcam (Real-time)
To execute tracking using your default webcam (source index `0`), run:
```bash
python main.py --source 0
```

### 2. Tracking on a Video File
Place your raw video inside the `input_videos/` folder, and pass its path to the script:
```bash
python main.py --source input_videos/traffic.mp4
```

### 3. Command Line Arguments
Custom configurations can be supplied directly from the terminal:
```bash
python main.py --source input_videos/traffic.mp4 --model models/yolov8s.pt --conf 0.40 --max-age 7 --min-hits 3
```

#### Available Arguments:
* `--source`: Input source (camera ID number, e.g. `0` or path to video file).
* `--model`: Path to YOLOv8 model weights (default: `models/yolov8n.pt`).
* `--conf`: Confidence threshold for detections (default: `0.35`).
* `--save`: Saves the output tracking video to `output_videos/` (default: enabled).
* `--no-show`: Disables the OpenCV pop-up frame display window (useful for headless servers).
* `--max-age`: Number of consecutive frames to keep a track alive without new updates (default: `5`).
* `--min-hits`: Number of consecutive frame detections required before publishing a track (default: `3`).
* `--iou-thresh`: Minimum IoU threshold to consider an association valid (default: `0.3`).

---

## 🖥️ Keyboard Controls (When Video Window is Active)
* **`p`**: Pause / Resume the execution stream.
* **`q`**: Quit the execution and close all windows.

---

## 📊 Sample Screenshots & Results
*(Add your screenshots of the tracked results in the `screenshots/` directory and embed them below to demonstrate tracking capability for your internship submission!)*

![Tracked Output Sample 1](screenshots/sample_tracking.jpg)

---

## 🔮 Future Scope / Extensions
- **Deep SORT Integration**: Adding a Convolutional Neural Network (CNN) feature extractor to incorporate visual appearance features, mitigating ID-switching under severe occlusions.
- **Entry & Exit Counting Lines**: Drawing horizontal/vertical counting lines on the screen to track exactly how many cars/people cross a specific border.
- **Speed Estimation**: Calculating velocity of targets based on homography metrics and scaling pixel movements.
