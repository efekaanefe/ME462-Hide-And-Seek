import cv2
import os
import pickle
from deepface import DeepFace

# === Settings ===
FOLDER = "known_faces_deepface"
ENCODING_FILE = "known_faces_deepface.pkl"
MODEL_NAME = "ArcFace"

# === Ask for person name ===
name = input("👤 Enter the person's name (no spaces): ").strip()
if not name:
    print("❌ Invalid name.")
    exit()

# === Create person's folder ===
person_folder = os.path.join(FOLDER, name)
os.makedirs(person_folder, exist_ok=True)

# === Capture from webcam ===
cap = cv2.VideoCapture(0)
print("📸 Press SPACE to capture a face image. Press ESC when done.")

img_count = 1
captured_paths = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Webcam read failed.")
        break

    cv2.imshow("Capture Face", frame)
    key = cv2.waitKey(1)

    if key % 256 == 27:  # ESC
        print("✅ Done capturing.")
        break

    elif key % 256 == 32:  # SPACE
        save_path = os.path.join(person_folder, f"{img_count}.jpg")
        cv2.imwrite(save_path, frame)
        captured_paths.append(save_path)
        print(f"✅ Saved image {img_count} at {save_path}")
        img_count += 1

cap.release()
cv2.destroyAllWindows()

if not captured_paths:
    print("❗ No images captured.")
    exit()

# === Load or initialize database ===
if os.path.exists(ENCODING_FILE):
    with open(ENCODING_FILE, "rb") as f:
        database = pickle.load(f)
else:
    database = {}

database[name] = []

# === Encode captured photos ===
print(f"🧠 Encoding {len(captured_paths)} image(s) for {name} using {MODEL_NAME}...")

for img_path in captured_paths:
    try:
        reps = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            enforce_detection=False,
        )
        if reps and isinstance(reps, list):
            emb = reps[0]["embedding"]
            database[name].append(emb)
            print(f"[+] Encoded: {os.path.basename(img_path)}")
        else:
            print(f"[!] No embedding for {img_path}")
    except Exception as e:
        print(f"[!] Error processing {img_path}: {e}")

# === Save to file ===
with open(ENCODING_FILE, "wb") as f:
    pickle.dump(database, f)

print(f"\n✅ Updated embeddings saved to {ENCODING_FILE}")
