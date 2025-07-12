# utils.py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # automatically downloads on first run

def detect_people(frame):
    results = model(frame)
    people = []
    for box in results[0].boxes:
        if int(box.cls[0]) == 0:  # class 0 = person
            people.append(box)
    return people
