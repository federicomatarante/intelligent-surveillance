from PIL import Image
from ultralytics import YOLO
import cv2

# Load a pretrained YOLOv8 model
model = YOLO('yolov8m.pt')

# Train the model
results = model.train(data="/Users/serenatrovalusci/Documents/UNI/CV_project_2/YOLO/virat_dataset.yaml", epochs=10)

print(results)

# Validate the model
metrics = model.val()
print(metrics.box.map)  # map50-95

