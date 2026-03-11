import cv2
from ultralytics import YOLO

# Load YOLOv8 model (downloads automatically if not present)
model = YOLO("yolov8s.pt")

# Vehicle classes from COCO dataset
vehicle_classes = [2, 3, 5, 7]  
# 2=car, 3=motorcycle, 5=bus, 7=truck

def detect_vehicles(img):

    results = model(img)[0]

    total = 0

    for box in results.boxes:

        cls = int(box.cls[0])

        if cls in vehicle_classes:

            total += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

    return img, total
