import os
import cv2
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(PROJECT_DIR, 'out')
os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
model = YOLO("yolov8s.pt")

CONF_THRESHOLD = 0.5

out = None
recording = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    people_count = 0
    PERSON_CLASS_ID = 0

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls[0])

            if cls == PERSON_CLASS_ID:
                people_count += 1

                x, y, w, h = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x,y), (w,h), (255,0,0), 2)

    if people_count > 0:
        cv2.circle(frame, (50, 50), 20, (0,255,0), -1)

        if not recording:
            print("START RECORDING")
            filename = os.path.join(OUT_DIR, f"video_{int(time.time())}.mp4")
            oyt = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, oyt, 20.0, (frame.shape[1], frame.shape[0]))
            recording = True

    else:
        cv2.circle(frame, (50, 50), 20, (0,0,255), -1)

        if recording:
            print("STOP RECORDING")
            out.release()
            recording = False


    if recording:
        out.write(frame)
    cv2.imshow("detection and caption", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
