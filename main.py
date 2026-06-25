import os
import time
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Sort

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
DEFAULT_TARGET_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck

def get_color(track_id):
    """
    Generates a unique, deterministic BGR color for each tracking ID.
    This ensures that a tracked object retains the same color across frames.
    """
    np.random.seed(int(track_id))
    # We generate numbers in range [50, 220] to avoid colors that are too dark or too bright
    color = tuple(map(int, np.random.randint(50, 220, size=3)))
    return color

def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Object Detection and Tracking using YOLOv8 & SORT")
    parser.add_argument(
        "--source", 
        type=str, 
        default="0", 
        help="Input video source: '0' for default webcam, or path to a video file"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="models/yolov8n.pt", 
        help="Path to YOLOv8 model weights (will download if not present)"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.35, 
        help="Confidence threshold for YOLO detections"
    )
    parser.add_argument(
        "--save", 
        action="store_true", 
        default=True,
        help="Save the processed tracked video to the output folder"
    )
    parser.add_argument(
        "--no-show", 
        action="store_true", 
        help="Do not display the video output window (useful for headless/automated testing)"
    )
    parser.add_argument(
        "--max-age", 
        type=int, 
        default=5, 
        help="Frames to keep a tracker active without updates"
    )
    parser.add_argument(
        "--min-hits", 
        type=int, 
        default=3, 
        help="Minimum consecutive frame updates before publishing a track"
    )
    parser.add_argument(
        "--iou-thresh", 
        type=float, 
        default=0.3, 
        help="IoU threshold for associating detections to tracks"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Initialize YOLOv8 Model
    print(f"[INFO] Loading YOLOv8 model from {args.model}...")
    # Ensure models directory exists
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    model = YOLO(args.model)
    print("[INFO] Model loaded successfully.")

    # 2. Initialize the SORT Tracker
    tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_thresh)
    print(f"[INFO] SORT Tracker initialized (max_age={args.max_age}, min_hits={args.min_hits}, iou_threshold={args.iou_thresh}).")

    # 3. Initialize Video Input Source
    source = args.source
    # Convert '0' or '1' string index to integer for Webcam
    if source.isdigit():
        source = int(source)
        print(f"[INFO] Opening Webcam source: {source}")
    else:
        if not os.path.exists(source):
            print(f"[ERROR] Input video path '{source}' does not exist.")
            return
        print(f"[INFO] Opening Video file source: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Failed to open input video source.")
        return

    # 4. Configure Video Output Settings
    save_output = args.save
    video_writer = None
    if save_output:
        os.makedirs("output_videos", exist_ok=True)
        if isinstance(source, int):
            out_filename = "webcam_tracked.mp4"
        else:
            base_name = os.path.basename(source)
            name_part, _ = os.path.splitext(base_name)
            out_filename = f"{name_part}_tracked.mp4"
            
        out_path = os.path.join("output_videos", out_filename)
        
        # Fetch resolution and FPS of input video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Default to 30 FPS if camera returns 0 or invalid FPS
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
            
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"[INFO] Output tracked video will be saved to: {out_path} at {fps} FPS, resolution: {width}x{height}")

    print("[INFO] Starting processing. Press 'q' to quit, 'p' to pause/resume.")
    
    prev_time = 0
    paused = False
    
    while True:
        if paused:
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = False
                print("[INFO] Resumed.")
            continue
            
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video file processing completed or stream ended.")
            break

        # 5. Run YOLOv8 object detection on current frame
        # verbose=False reduces terminal clutter during execution
        results = model.predict(frame, conf=args.conf, verbose=False)
        
        # 6. Parse detection results into SORT-compatible format
        # Format: [ [x1, y1, x2, y2, confidence, class_id], ... ]
        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls_id = int(box.cls[0])
                
                # Filter to only keep our target classes (e.g. Person, Vehicle)
                if cls_id in DEFAULT_TARGET_CLASSES:
                    detections.append([x1, y1, x2, y2, conf, cls_id])

        # If no target classes were detected, pass empty array of correct shape
        if len(detections) == 0:
            dets_np = np.empty((0, 6))
        else:
            dets_np = np.array(detections)

        # 7. Update tracking positions using the SORT algorithm
        # Returns: numpy array where each row represents: [x1, y1, x2, y2, track_id, class_id, confidence]
        tracked_objects = tracker.update(dets_np)

        # 8. Render bounding boxes and statistics on the frame
        active_counts = {}
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id, cls_id, conf = obj
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id = int(track_id)
            cls_id = int(cls_id)
            
            # Map class ID to readable label
            cls_name = COCO_CLASSES.get(cls_id, 'unknown')
            
            # Increment class count for frame summary
            active_counts[cls_name] = active_counts.get(cls_name, 0) + 1
            
            # Dynamic bounding box color based on track_id
            color = get_color(track_id)
            
            # Draw primary bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID and Label on top of bounding box
            label = f"{cls_name.upper()} #{track_id} ({conf:.0%})"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw semi-transparent background for label to ensure legibility
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Calculate Frame Rate (FPS)
        current_time = time.time()
        fps_text = f"FPS: {1.0 / (current_time - start_time):.1f}"
        
        # Render clean dashboard overlay at the top left of the frame
        cv2.rectangle(frame, (10, 10), (320, 80 + len(active_counts) * 20), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (320, 80 + len(active_counts) * 20), (200, 200, 200), 1)
        
        cv2.putText(frame, "Object Tracking Dashboard", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, fps_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        
        total_tracked = len(tracked_objects)
        cv2.putText(frame, f"Total Active Tracks: {total_tracked}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display breakdown of detected object categories
        y_offset = 90
        for name, count in active_counts.items():
            cv2.putText(frame, f"- {name.capitalize()}s: {count}", (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            y_offset += 20

        # 9. Write processed frame to output file
        if video_writer is not None:
            video_writer.write(frame)

        # 10. Display the resulting frame
        if not args.no_show:
            cv2.imshow("Real-Time Object Detection & Tracking System", frame)
            
            # Key bindings for interaction
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = True
                print("[INFO] Paused. Press 'p' to resume.")

    # Cleanup and release resources
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Processing finished. Resources released.")

if __name__ == "__main__":
    main()
