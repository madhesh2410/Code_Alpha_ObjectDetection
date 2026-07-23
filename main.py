import cv2
import numpy as np
from ultralytics import YOLO
from tracker.sort import Sort

# Load YOLO model
model = YOLO("yolov8m.pt")

# Initialize SORT tracker
tracker = Sort()

# Open webcam
camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        break

    # Run YOLO detection
    results = model(frame)

    detections = []
    detection_info = []

    # Extract detections
    for result in results:
        for box in result.boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Higher confidence = fewer false detections
            if confidence > 0.60:
                detections.append([x1, y1, x2, y2, confidence])

                detection_info.append({
                    "box": [x1, y1, x2, y2],
                    "class": class_name,
                    "confidence": confidence
                })

    detections = np.array(detections)

    if len(detections) > 0:
        tracked_objects = tracker.update(detections)
    else:
        tracked_objects = tracker.update(np.empty((0, 5)))

    # Draw tracking results
    for track in tracked_objects:

        x1, y1, x2, y2, track_id = track.astype(int)

        best_class = "Object"
        best_conf = 0
        best_iou = 0

        # Match SORT box with YOLO detection
        for det in detection_info:

            dx1, dy1, dx2, dy2 = det["box"]

            # Calculate IoU
            xx1 = max(x1, dx1)
            yy1 = max(y1, dy1)
            xx2 = min(x2, dx2)
            yy2 = min(y2, dy2)

            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)

            area1 = (x2 - x1) * (y2 - y1)
            area2 = (dx2 - dx1) * (dy2 - dy1)

            union = area1 + area2 - inter

            iou = inter / union if union > 0 else 0

            if iou > best_iou:
                best_iou = iou
                best_class = det["class"]
                best_conf = det["confidence"]

        label = f"{best_class} | ID:{track_id} | {best_conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    cv2.imshow("YOLO + SORT Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()