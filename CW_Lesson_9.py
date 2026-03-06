import cv2
import numpy as np
import os

eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
dnn = cv2.dnn.readNetFromCaffe('dnl/deploy.prototxt', 'dnl/res10_300x300_ssd_iter_140000.caffemodel')

frame = cv2.imread('image/smith1.png')

(h, w) = frame.shape[:2]
blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
dnn.setInput(blob)
detections = dnn.forward()

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x, y, y2, x2) = box.astype("int")
        x, y = max(0, x), max(0, y)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
eyes = eye_cascade.detectMultiScale(frame, scaleFactor=1.1 ,minNeighbors=10, minSize=(10, 10))
for (x, y, w, h) in eyes:
    for(x, y, w, h) in eyes:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255 ,0, 0 ), 1)



cv2.imshow('jef', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output/result.png', frame)

input_folder = "image"
output_folder = "output"

formats = ('.jpg', '.jpeg', '.png', 'webp')

os.makedirs(output_folder, exist_ok=True)
files = sorted(os.listdir(input_folder))


for file in files:
    if not file.lower().endswith(formats):
        continue

    path = os.path.join(input_folder, file)
    img = cv2.imread(path)
    if img is None:
        continue
    output_path = os.path.join(output_folder, file)

    cv2.imwrite(output_path, frame)
