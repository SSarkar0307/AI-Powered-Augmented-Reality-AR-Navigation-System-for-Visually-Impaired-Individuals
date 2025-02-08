
### DETECTS OBSTACLES AND VOICE NAVIGATION ASSISTANCE LIVE ###

from ultralytics import YOLO
import cv2
import torch
import pyttsx3
import time
import queue
from threading import Thread

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust voice speed

# Speech queue and thread setup
speech_queue = queue.Queue()
speech_thread_running = True

def speech_worker():
    """Thread function to process speech commands without blocking."""
    while speech_thread_running:
        try:
            text = speech_queue.get(timeout=0.1)
            if text:  # Ensure it's a valid command
                engine.say(text)
                engine.runAndWait()  # Blocking, but in its own thread
            speech_queue.task_done()
        except queue.Empty:
            continue

# Start speech thread
speech_thread = Thread(target=speech_worker, daemon=True)
speech_thread.start()

# Ensure CUDA (GPU) is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLOv8 model and use half-precision for better performance
model = YOLO("yolov8n.pt").to(device)
if device == "cuda":
    model.model.half()

# Set up video capture with reduced resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Load YOLO class names
class_names = model.names

# Define frame regions
FRAME_WIDTH = 640
REGIONS = {
    "left": (0, 213),
    "center": (214, 426),
    "right": (427, 640)
}
MIN_BOX_HEIGHT = 100  # Minimum height for considering objects

# Track last spoken times
last_speak_time = time.time()
last_command = ""
move_forward_delay = 2  # 2 seconds between "Move Forward"
urgent_command_delay = 1  # 1 second for urgent commands

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB and process
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO inference with half-precision
    with torch.no_grad():
        results = model(frame_rgb, half=(device == "cuda"), verbose=False)

    # Process detections
    detections = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    
    blocked_regions = {"left": False, "center": False, "right": False}
    object_name = "Obstacle"

    for box, cls in zip(detections, classes):
        x1, y1, x2, y2 = box
        height = y2 - y1
        if height < MIN_BOX_HEIGHT:
            continue

        center_x = (x1 + x2) / 2
        obj_name = class_names[int(cls)]
        object_name = obj_name  # Track last detected object

        # Check regions
        for region, (start, end) in REGIONS.items():
            if start <= center_x <= end:
                blocked_regions[region] = True

    # Determine movement suggestion
    if blocked_regions["center"]:
        if not blocked_regions["left"]:
            cmd = f"Move Left, obstacle ahead"
        elif not blocked_regions["right"]:
            cmd = f"Move Right, obstacle ahead"
        else:
            cmd = f"Stop! obstacle blocking path"
    else:
        cmd = "Move Forward"

    # Rate-limit voice commands
    current_time = time.time()
    delay = move_forward_delay if cmd == "Move Forward" else urgent_command_delay
    if cmd != last_command and (current_time - last_speak_time > delay):
        if speech_queue.empty():  # Avoid piling up commands
            speech_queue.put(cmd)
        last_speak_time = current_time
        last_command = cmd

    # Display annotations
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, cmd, (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow("Navigation Assistance", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
speech_thread_running = False
speech_thread.join()
cap.release()
cv2.destroyAllWindows()
