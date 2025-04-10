import cv2
import os
import face_recognition
import pickle

FOLDER = "known_faces_dlib"
ENCODINGS_FILE = "known_faces_dlib.pkl"

# Ensure main folder exists
os.makedirs(FOLDER, exist_ok=True)

# Ask for the person's name
name = input("👤 Enter the person's name (no spaces): ").strip()
if not name:
    print("❌ Invalid name.")
    exit()

# Create person's subfolder
person_folder = os.path.join(FOLDER, name)
os.makedirs(person_folder, exist_ok=True)

# Load encodings if available and make sure it's a dict
known_faces = {}
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
        if isinstance(data, dict):
            known_faces = data
        else:
            print("⚠️ Invalid encoding format detected. Resetting known_faces.pkl.")
            os.remove(ENCODINGS_FILE)
            known_faces = {}

# Ensure name exists in dict
if name not in known_faces:
    known_faces[name] = []

# Start webcam
cap = cv2.VideoCapture(0)
print("📸 Press SPACE to capture a face image. Press ESC when done.")

img_count = 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read from webcam.")
        break

    cv2.imshow("New Face Capture", frame)
    key = cv2.waitKey(1)

    if key % 256 == 27:  # ESC
        print("✅ Done capturing images.")
        break
    elif key % 256 == 32:  # SPACE
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)

        if not encodings:
            print("❌ No face found. Try again.")
            continue

        # Save image
        save_path = os.path.join(person_folder, f"{img_count}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"✅ Image {img_count} saved as {save_path}")

        # Save encoding
        known_faces[name].append(encodings[0])
        img_count += 1

cap.release()
cv2.destroyAllWindows()

# Save encodings to file
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(known_faces, f)

print("✅ Encodings saved to known_faces.pkl.")
