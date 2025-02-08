
### DETECTS OBSTACLES AND MOVEMENT SUGGESTION IN VIDEO FILES ###

from ultralytics import YOLO
import cv2
import torch

# Ensure CUDA (GPU) is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLOv8 model
model = YOLO("yolov8n.pt").to(device)

# Set up video capture from file
video_path = "input_video.mp4"  # Change this to your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

print("Video file loaded successfully!")

# Load YOLO class names
class_names = model.names  # Dictionary mapping class IDs to names

# Define frame regions
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
LEFT_REGION = (0, FRAME_WIDTH // 3)  # Left third
CENTER_REGION = (FRAME_WIDTH // 3, 2 * FRAME_WIDTH // 3)  # Middle third
RIGHT_REGION = (2 * FRAME_WIDTH // 3, FRAME_WIDTH)  # Right third
MIN_BOX_HEIGHT = 100  # Tune based on real testing

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # Resize the frame (keep aspect ratio)
    frame_resized = cv2.resize(frame, (720, 600))

    # Run YOLO object detection
    results = model(frame_resized)

    # Get detections from YOLO result
    detections = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    classes = results[0].boxes.cls.cpu().numpy()  # Class labels

    # Flags to check if regions are blocked
    left_blocked = False
    center_blocked = False
    right_blocked = False
    object_name = ""

    for box, cls in zip(detections, classes):
        x1, y1, x2, y2 = box
        height = y2 - y1  # Bounding box height (used for distance approximation)
        center_x = (x1 + x2) / 2

        if height > MIN_BOX_HEIGHT:  # Only consider large objects (closer)
            obj_name = class_names[int(cls)]  # Get object name from class ID
            if CENTER_REGION[0] <= center_x <= CENTER_REGION[1]:
                center_blocked = True
                object_name = obj_name
            elif LEFT_REGION[0] <= center_x <= LEFT_REGION[1]:
                left_blocked = True
            elif RIGHT_REGION[0] <= center_x <= RIGHT_REGION[1]:
                right_blocked = True

    # Decide movement suggestion
    if center_blocked:
        if not left_blocked:
            movement_suggestion = f" Move Left, obstacle Ahead"
        elif not right_blocked:
            movement_suggestion = f" Move Right, obstacle Ahead"
        else:
            movement_suggestion = f" Stop! obstacle Blocking Path"
    else:
        movement_suggestion = " Move Forward"

    # Draw detected objects
    annotated_frame = results[0].plot()

    # Show movement suggestion
    cv2.putText(annotated_frame, movement_suggestion, (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display frame
    cv2.imshow("Navigation Assistance (Video)", annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
