import cv2
import insightface
import pickle
import numpy as np
import os

# === Suppress logs ===
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["INSIGHTFACE_LOG_LEVEL"] = "ERROR"

# === Load known embeddings ===
with open("known_faces_arcface.pkl", "rb") as f:
    raw_faces = pickle.load(f)

known_names = []
known_embeddings = []

for name, embeddings in raw_faces.items():
    for emb in embeddings:
        norm_emb = emb / np.linalg.norm(emb)
        known_names.append(name)
        known_embeddings.append(norm_emb)

# === Init InsightFace for detection + embedding ===
model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0)  # GPU=0, CPU=-1

# === Webcam feed ===
cap = cv2.VideoCapture(0)
print("🎥 Running ArcFace Face Recognition... (Press ESC to exit)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    faces = model.get(frame)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        embedding = face.embedding
        embedding = embedding / np.linalg.norm(embedding)

        name = "Unknown"
        similarities = [np.dot(embedding, k) for k in known_embeddings]
        best_index = int(np.argmax(similarities))
        best_score = similarities[best_index]

        if best_score > 0.36:
            name = known_names[best_index]

        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("ArcFace Recognition", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
