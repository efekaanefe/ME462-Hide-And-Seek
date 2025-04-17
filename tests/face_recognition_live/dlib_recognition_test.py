import cv2
import mediapipe as mp
import face_recognition
import pickle

# === Load known face encodings ===
with open("known_faces_dlib.pkl", "rb") as f:
    known_faces = pickle.load(f)

# Flatten encodings and match names
known_encodings = []
known_names = []

for name, enc_list in known_faces.items():
    for enc in enc_list:
        known_encodings.append(enc)
        known_names.append(name)

# === Initialize MediaPipe Face Detection ===
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# === Start Webcam ===
cap = cv2.VideoCapture(0)
print("🎥 Starting real-time face recognition...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Ensure bounding box is within frame bounds
            x = max(0, x)
            y = max(0, y)
            width = min(w - x, width)
            height = min(h - y, height)

            # Convert to face_recognition format (top, right, bottom, left)
            face_location = (y, x + width, y + height, x)

            name = "Unknown"
            encodings = face_recognition.face_encodings(rgb, known_face_locations=[face_location])
            if encodings:
                encoding = encodings[0]
                matches = face_recognition.compare_faces(known_encodings, encoding)
                face_distances = face_recognition.face_distance(known_encodings, encoding)

                if any(matches):
                    best_match_index = face_distances.argmin()
                    name = known_names[best_match_index]

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
