
import cv2
import time
import numpy as np
from mtcnn import MTCNN
import mediapipe as mp

# Load image
img = cv2.imread('your_face_image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 1. Haar Cascade
haar_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
start = time.time()
haar_faces = haar_face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
print(f"Haar Cascade Time: {time.time() - start:.3f}s, Detected: {len(haar_faces)} faces")

# 2. MTCNN
detector = MTCNN()
start = time.time()
mtcnn_faces = detector.detect_faces(img_rgb)
print(f"MTCNN Time: {time.time() - start:.3f}s, Detected: {len(mtcnn_faces)} faces")

# 3. Mediapipe
mp_face_detection = mp.solutions.face_detection
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    start = time.time()
    result = face_detection.process(img_rgb)
    faces = result.detections if result.detections else []
    print(f"Mediapipe Time: {time.time() - start:.3f}s, Detected: {len(faces)} faces")

# Optional: visualize results
for (x, y, w, h) in haar_faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
for face in mtcnn_faces:
    x, y, w, h = face['box']
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
if result.detections:
    for detection in result.detections:
        bbox = detection.location_data.relative_bounding_box
        ih, iw, _ = img.shape
        x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow('Comparison', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
