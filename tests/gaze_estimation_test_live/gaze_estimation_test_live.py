import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Webcam
cap = cv2.VideoCapture(0)

# Define 3D model points of face landmarks (reference points in 3D space)
# These are approximate coordinates (standardized face model)
face_3d_model = np.array([
    (0.0, 0.0, 0.0),         # Nose tip
    (0.0, -63.6, -12.5),     # Chin
    (-43.3, 32.7, -26.0),    # Left eye left corner
    (43.3, 32.7, -26.0),     # Right eye right corner
    (-28.9, -28.9, -24.1),   # Left mouth corner
    (28.9, -28.9, -24.1)     # Right mouth corner
], dtype=np.float64)

# 2D landmark indices corresponding to those points
landmark_ids = [1, 152, 263, 33, 287, 57]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_h, img_w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # 2D image points
        face_2d = []
        for idx in landmark_ids:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            # Draw landmarks
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        face_2d = np.array(face_2d, dtype=np.float64)

        # Camera internals
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4,1))  # No distortion

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            face_3d_model, face_2d, camera_matrix, dist_coeffs
        )

        # Get rotational matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vector)

        # Get angles
        proj_matrix = np.hstack((rotation_mat, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch, yaw, roll = euler_angles.flatten()

        # Display rotation angles
        text = f"Yaw: {int(yaw)}, Pitch: {int(pitch)}, Roll: {int(roll)}"
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display head direction roughly
        if yaw < -15:
            cv2.putText(frame, "Looking Left", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif yaw > 15:
            cv2.putText(frame, "Looking Right", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif pitch < -10:
            cv2.putText(frame, "Looking Up", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif pitch > 10:
            cv2.putText(frame, "Looking Down", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Looking Forward", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Head Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
