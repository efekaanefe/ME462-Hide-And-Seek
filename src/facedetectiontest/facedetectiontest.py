import cv2
import time
import mediapipe as mp
from mtcnn import MTCNN
import numpy as np
import dlib
import face_recognition

# Load the image
image_path = "face_image4.jpg"  # Change this to your image path
image = cv2.imread(image_path)

# Convert to grayscale for Haar Cascade and Dlib
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Haar Cascade face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
start_time = time.time()
faces_haar = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
end_time = time.time()
haar_time = end_time - start_time
haar_count = len(faces_haar)

# MTCNN face detection
mtcnn_detector = MTCNN()
start_time = time.time()
faces_mtcnn = mtcnn_detector.detect_faces(image)
end_time = time.time()
mtcnn_time = end_time - start_time
mtcnn_count = len(faces_mtcnn)

# MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mediapipe_count = 0

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    start_time = time.time()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    end_time = time.time()
    mediapipe_time = end_time - start_time
    if results.detections:
        mediapipe_count = len(results.detections)

# Dlib face detection
dlib_detector = dlib.get_frontal_face_detector()
start_time = time.time()
dlib_faces = dlib_detector(gray)
end_time = time.time()
dlib_time = end_time - start_time
dlib_count = len(dlib_faces)

# Face Recognition library face detection
start_time = time.time()
face_locations = face_recognition.face_locations(image)
end_time = time.time()
face_recognition_time = end_time - start_time
face_recognition_count = len(face_locations)

# Display the results with color information
print("Face Detection Results:")
print("Haar Cascade - Time: {:.4f} sec, Faces: {}, Color: Blue".format(haar_time, haar_count))
print("MTCNN - Time: {:.4f} sec, Faces: {}, Color: Green".format(mtcnn_time, mtcnn_count))
print("MediaPipe - Time: {:.4f} sec, Faces: {}, Color: Red".format(mediapipe_time, mediapipe_count))
print("Dlib - Time: {:.4f} sec, Faces: {}, Color: Cyan".format(dlib_time, dlib_count))
print("Face Recognition - Time: {:.4f} sec, Faces: {}, Color: Yellow".format(face_recognition_time, face_recognition_count))

# Draw rectangles for visualization
for (x, y, w, h) in faces_haar:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

for face in faces_mtcnn:
    x, y, w, h = face['box']
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

if results.detections:
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = image.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

for face in dlib_faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

for (top, right, bottom, left) in face_locations:
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 255), 2)

cv2.imshow('Face Detection Comparison', image)
cv2.waitKey(0)
cv2.destroyAllWindows()