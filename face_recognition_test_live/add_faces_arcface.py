import cv2
import os
import pickle
import numpy as np
import insightface

FOLDER = "known_faces_arcface"
ENCODINGS_FILE = "known_faces_arcface.pkl"

# Ensure folder exists
os.makedirs(FOLDER, exist_ok=True)

# Ask for the person's name
name = input("👤 Enter the person's name (no spaces): ").strip()
if not name:
    print("❌ Invalid name.")
    exit()

# Create person's folder
person_folder = os.path.join(FOLDER, name)
os.makedirs(person_folder, exist_ok=True)

# Load or initialize known faces
known_faces = {}
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
        if isinstance(data, dict):
            known_faces = data
        else:
            print("⚠️ Invalid encoding format detected. Resetting.")
            os.remove(ENCODINGS_FILE)
            known_faces = {}

# Ensure person entry exists
if name not in known_faces:
    known_faces[name] = []

# Prepare ArcFace model
model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0)

# Webcam capture loop
cap = cv2.VideoCapture(0)
print("📸 Press SPACE to capture a face image. Press ESC when done.")

img_count = 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Webcam read failed.")
        break

    # Detect face and draw boxes (optional)
    faces = model.get(frame)
    for face in faces:
        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    cv2.imshow("ArcFace Capture", frame)
    key = cv2.waitKey(1)

    if key % 256 == 27:  # ESC
        print("✅ Done capturing.")
        break
    elif key % 256 == 32:  # SPACE
        faces = model.get(frame)
        if not faces:
            print("❌ No face detected. Try again.")
            continue

        embedding = faces[0].embedding
        embedding = embedding / np.linalg.norm(embedding)  # Normalize

        # Save image
        save_path = os.path.join(person_folder, f"{img_count}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"✅ Saved {save_path}")

        # Save embedding
        known_faces[name].append(embedding)
        img_count += 1

cap.release()
cv2.destroyAllWindows()

# Save updated encodings
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(known_faces, f)

print(f"✅ All encodings saved to {ENCODINGS_FILE}")
