
### DETECTS CENTER BLOCKING OBSTACLES ONLY LIVE ###

from ultralytics import YOLO
import cv2
import torch

# Ensure CUDA (GPU) is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLOv8 model on GPU (or CPU if not available)
model = YOLO("yolov8n.pt").to(device)  # 'n' for speed

# Set up video capture (use phone IP stream here)
cap = cv2.VideoCapture(0)  # Use VideoCapture for the phone camera

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Unable to open video stream.")
    exit()

print("Video stream opened successfully!")

# Reduce resolution for better speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Load YOLO class names
class_names = model.names  # Dictionary mapping class IDs to names

# Thresholds for distance warnings (calibrated based on object size)
MIN_BOX_HEIGHT = 100  # Tune based on real testing
CENTER_REGION_START = 200
CENTER_REGION_END = 440  # Defines the central part of the frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Resize the frame
    frame_resized = cv2.resize(frame, (640, 640))

    # Run YOLO object detection
    results = model(frame_resized)

    # Get detections from YOLO result
    detections = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates (x1, y1, x2, y2)
    classes = results[0].boxes.cls.cpu().numpy()  # Class labels

    warning_message = None

    for box, cls in zip(detections, classes):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1  # Bounding box height (used for distance approximation)

        # Check if the object is in the center region
        center_x = (x1 + x2) / 2
        if CENTER_REGION_START < center_x < CENTER_REGION_END and height > MIN_BOX_HEIGHT:
            object_name = class_names[int(cls)]  # Get object name from class ID
            warning_message = f"⚠️ {object_name} in front!"
            break  # One object is enough to trigger a warning

    # Draw detected objects
    annotated_frame = results[0].plot()

    # Show warning on frame
    if warning_message:
        cv2.putText(annotated_frame, warning_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display frame
    cv2.imshow("Live Object Detection", annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
