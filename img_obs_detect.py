
### DETECTS OBJECTS IN IMAGE ###

from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # 'n' means nano version (small and fast)

# Load an image
image_path = "test.jpg"  # Replace with your image path
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
    exit()

# Run YOLO object detection
results = model(image)

# Plot detections
annotated_image = results[0].plot()

# Resize the window to fit properly
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)  # Enable resizing
cv2.imshow("Detected Objects", annotated_image)

# Allow user to manually resize window and view the image
cv2.waitKey(0)
cv2.destroyAllWindows()

