import os
import tempfile
import cv2
import numpy as np
import streamlit as st
import time
import pandas as pd
from ultralytics import YOLO
from tracker import Sort

# Set page configuration
st.set_page_config(
    page_title="Real-Time Object Detection & Tracking",
    page_icon="🎥",
    layout="wide"
)

# Title & Description
st.title("🎥 AI Object Detection & Tracking System")
st.markdown("""
Welcome to the **Real-Time Object Detection and Tracking Dashboard**. 
This application utilizes a pre-trained **YOLOv8** model for object detection and a custom **SORT (Simple Online and Realtime Tracking)** tracker to identify and trace objects across video frames.
""")

# COCO dataset class index to name mapping (standard for pre-trained YOLOv8)
COCO_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck'
}

# Targeted classes for our tracking system
TRACKING_CLASSES = {
    'Person': 0,
    'Bicycle': 1,
    'Car': 2,
    'Motorcycle': 3,
    'Bus': 5,
    'Truck': 7
}

def get_color(track_id):
    """
    Generates a unique, deterministic BGR color for each tracking ID.
    """
    np.random.seed(int(track_id))
    color = tuple(map(int, np.random.randint(50, 220, size=3)))
    return color

# Sidebar Configuration Panel
st.sidebar.header("🛠️ Configuration Panel")

# 1. Model Selection
model_size = st.sidebar.selectbox(
    "Select YOLOv8 Model Size",
    ["YOLOv8 Nano (Fastest)", "YOLOv8 Small (Balanced)", "YOLOv8 Medium (Accurate)"],
    index=0
)
model_mapping = {
    "YOLOv8 Nano (Fastest)": "models/yolov8n.pt",
    "YOLOv8 Small (Balanced)": "models/yolov8s.pt",
    "YOLOv8 Medium (Accurate)": "models/yolov8m.pt"
}
model_path = model_mapping[model_size]

# 2. Input Source Selection
source_type = st.sidebar.radio("Select Input Source", ["Upload Video File", "Webcam Input"])

uploaded_file = None
if source_type == "Upload Video File":
    uploaded_file = st.sidebar.file_uploader("Upload Video (mp4, avi, mov, mkv)", type=["mp4", "avi", "mov", "mkv"])
else:
    webcam_index = st.sidebar.number_input("Webcam Index", min_value=0, max_value=10, value=0, step=1)

# 3. Class Filters
selected_class_names = st.sidebar.multiselect(
    "Select Classes to Detect & Track",
    list(TRACKING_CLASSES.keys()),
    default=list(TRACKING_CLASSES.keys())
)
selected_class_ids = [TRACKING_CLASSES[name] for name in selected_class_names]

# 4. Tracking Parameters
st.sidebar.subheader("⚙️ Tracking Parameters")
conf_threshold = st.sidebar.slider("YOLO Detection Confidence", 0.10, 1.00, 0.35, 0.05)
max_age = st.sidebar.slider("Max Age (Frames to keep inactive tracks)", 1, 15, 5, 1)
min_hits = st.sidebar.slider("Min Hits (Frames before publishing track)", 1, 10, 3, 1)
iou_thresh = st.sidebar.slider("IoU Association Threshold", 0.10, 0.90, 0.30, 0.05)

# Placeholders for status and preview
start_button = st.button("🚀 Start Object Tracking", disabled=(source_type == "Upload Video File" and uploaded_file is None))

if start_button:
    # Set up temp files for input/output
    tfile = None
    input_path = None
    
    if source_type == "Upload Video File" and uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        input_path = tfile.name
        st.info(f"Processing uploaded video file...")
    else:
        input_path = webcam_index
        st.info(f"Opening webcam feed (index: {webcam_index})...")

    # Load YOLO model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model = YOLO(model_path)
    
    # Initialize SORT tracker
    tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_thresh)
    
    # Open Video Source
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        st.error("Error: Could not open video source.")
    else:
        # Layout columns for real-time stats
        col1, col2, col3 = st.columns(3)
        stat_fps = col1.empty()
        stat_active_tracks = col2.empty()
        stat_total_counts = col3.empty()

        # Layout for frame rendering and breakdown list
        preview_col, stats_col = st.columns([2, 1])
        frame_placeholder = preview_col.empty()
        breakdown_placeholder = stats_col.empty()
        
        # Output Video Writer
        out_filename = "output_videos/streamlit_output.mp4"
        os.makedirs("output_videos", exist_ok=True)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))
        
        # Tracking history for analytics charts
        counts_history = []
        chart_placeholder = st.empty()
        
        frame_idx = 0
        
        # Stop tracking button
        stop_btn = st.checkbox("Stop Execution")

        while cap.isOpened() and not stop_btn:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            
            # YOLO detection
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls_id = int(box.cls[0])
                    
                    if cls_id in selected_class_ids:
                        detections.append([x1, y1, x2, y2, conf, cls_id])

            if len(detections) == 0:
                dets_np = np.empty((0, 6))
            else:
                dets_np = np.array(detections)

            # SORT tracking update
            tracked_objects = tracker.update(dets_np)
            
            # Draw overlay graphics
            active_counts = {}
            for obj in tracked_objects:
                x1, y1, x2, y2, track_id, cls_id, conf = obj
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                track_id = int(track_id)
                cls_id = int(cls_id)
                
                cls_name = COCO_CLASSES.get(cls_id, 'unknown')
                active_counts[cls_name] = active_counts.get(cls_name, 0) + 1
                
                color = get_color(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{cls_name.upper()} #{track_id} ({conf:.0%})"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Save frame to output video
            video_writer.write(frame)

            # Calculate FPS
            elapsed = time.time() - start_time
            fps_val = 1.0 / elapsed if elapsed > 0 else 0.0
            
            # Update metrics panels
            stat_fps.metric("Processing speed", f"{fps_val:.1f} FPS")
            stat_active_tracks.metric("Current Active Tracks", len(tracked_objects))
            stat_total_counts.metric("Frame Index", frame_idx)
            
            # Convert BGR frame to RGB for Streamlit rendering
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, use_column_width=True)
            
            # Format breakdown metrics
            breakdown_text = "### 📊 Category Count Breakdown\n"
            if len(active_counts) == 0:
                breakdown_text += "No targeted objects detected in this frame."
            else:
                for k, v in active_counts.items():
                    breakdown_text += f"- **{k.capitalize()}s**: {v}\n"
            breakdown_placeholder.markdown(breakdown_text)
            
            # Append breakdown stats for chart history
            active_counts['Frame'] = frame_idx
            counts_history.append(active_counts)
            
            # Display real-time line chart of tracked categories
            if frame_idx % 10 == 0 and len(counts_history) > 0:
                df_history = pd.DataFrame(counts_history).fillna(0).set_index('Frame')
                with chart_placeholder.container():
                    st.markdown("### 📈 Detection Counts Over Time")
                    st.line_chart(df_history)

        # Release resources
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        
        # Clean up temporary uploaded file if any
        if tfile is not None:
            try:
                os.unlink(input_path)
            except Exception:
                pass
                
        st.success("🎉 Processing Finished!")
        st.info(f"Processed video has been saved to: `{out_filename}`")
        
        # Provide direct file download button
        with open(out_filename, "rb") as file:
            st.download_button(
                label="📥 Download Processed Video",
                data=file,
                file_name="tracked_output.mp4",
                mime="video/mp4"
            )
