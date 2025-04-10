import cv2
import pickle
import numpy as np
from deepface import DeepFace

# === Settings ===
ENCODING_FILE = "known_faces_deepface.pkl"
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "mediapipe"  # Try: retinaface, ssd, mtcnn, mediapipe

# === Load known embeddings ===
print("📂 Loading known embeddings...")
with open(ENCODING_FILE, "rb") as f:
    known_faces = pickle.load(f)

# === Webcam feed ===
cap = cv2.VideoCapture(0)
print("🎥 Running DeepFace-only recognition (Press ESC to quit)...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Detect and align all faces (no target_size!)
        detected_faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False
        )

        for face_obj in detected_faces:
            face_img = face_obj.get("face")
            facial_area = face_obj.get("facial_area")

            if face_img is None or facial_area is None:
                continue

            # Facial area coordinates
            x = facial_area.get("x", 0)
            y = facial_area.get("y", 0)
            w = facial_area.get("w", 0)
            h = facial_area.get("h", 0)

            # Compute embedding
            rep = DeepFace.represent(
                img_path=face_img,
                model_name=MODEL_NAME,
                enforce_detection=False
            )[0]["embedding"]
            rep = np.array(rep)

            # Match against known embeddings
            best_match = "Unknown"
            best_score = -1

            for name, enc_list in known_faces.items():
                for known_rep in enc_list:
                    known_rep = np.array(known_rep)
                    sim = np.dot(rep, known_rep) / (np.linalg.norm(rep) * np.linalg.norm(known_rep))
                    if sim > best_score:
                        best_score = sim
                        best_match = name

            label = best_match if best_score > 0.36 else "Unknown"

            # Draw result
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    except Exception as e:
        print(f"[!] Error: {e}")

    cv2.imshow("DeepFace Recognition", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
