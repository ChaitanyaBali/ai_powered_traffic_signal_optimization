import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

def detect_vehicles(image):

    results = model(image)

    vehicle_count = 0

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Vehicle classes in YOLO
            if cls in [2, 3, 5, 7]:  # car, bike, bus, truck

                vehicle_count += 1

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

                # Label
                cv2.putText(
                    image,
                    "Vehicle",
                    (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2
                )

    return image, vehicle_count