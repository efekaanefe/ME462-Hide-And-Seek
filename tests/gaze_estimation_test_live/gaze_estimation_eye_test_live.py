import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # Needed to get iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye and iris landmark indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_CORNER = [33, 133]   # [Left corner, Right corner]
RIGHT_CORNER = [362, 263] # [Left corner, Right corner]
LEFT_LID = [159, 145]
RIGHT_LID = [386, 374]


# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame for natural (mirror) viewing
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            img_h, img_w, _ = frame.shape

            # Get iris center points
            left_iris = np.mean([(face_landmarks.landmark[i].x * img_w, face_landmarks.landmark[i].y * img_h) for i in LEFT_IRIS], axis=0)
            right_iris = np.mean([(face_landmarks.landmark[i].x * img_w, face_landmarks.landmark[i].y * img_h) for i in RIGHT_IRIS], axis=0)

            # Get eye corner points
            left_corner = [(face_landmarks.landmark[i].x * img_w, face_landmarks.landmark[i].y * img_h) for i in LEFT_CORNER]
            right_corner = [(face_landmarks.landmark[i].x * img_w, face_landmarks.landmark[i].y * img_h) for i in RIGHT_CORNER]

            # Get lid Points
            left_lid = [(face_landmarks.landmark[i].x * img_w, face_landmarks.landmark[i].y * img_h) for i in LEFT_LID]
            right_lid = [(face_landmarks.landmark[i].x * img_w, face_landmarks.landmark[i].y * img_h) for i in RIGHT_LID]

            # Compute horizontal gaze ratios
            left_eye_width = left_corner[1][0] - left_corner[0][0]
            right_eye_width = right_corner[1][0] - right_corner[0][0]

            left_eye_height = left_lid[1][1] - left_lid[0][1]
            right_eye_height = right_lid[1][1] - right_lid[0][1]

            left_horizontal_ratio = (left_iris[0] - left_corner[0][0]) / (left_eye_width)
            right_horizontal_ratio = (right_iris[0] - right_corner[0][0]) / (right_eye_width)

            left_vertical_ratio = (left_iris[1] - left_lid[0][1]) / (left_eye_height)
            right_vertical_ratio = (right_iris[1] - right_lid[0][1]) / (right_eye_height)

            horizontal_ratio = (left_horizontal_ratio + right_horizontal_ratio) / 2
            vertical_ratio = (left_vertical_ratio + right_vertical_ratio) / 2
            

            # Decision logic
            if horizontal_ratio <= 0.35:
                gaze_horizontal = "Looking Left"
            elif horizontal_ratio >= 0.65:
                gaze_horizontal = "Looking Right"
            else:
                gaze_horizontal = "Looking Center"
            print(vertical_ratio)
            # Decision logic
            if vertical_ratio <= 0.35:
                gaze_vertical = "Looking Up"
            elif vertical_ratio >= 0.65:
                gaze_vertical = "Looking Down"
            else:
                gaze_vertical = "Looking Center"

            # Draw eyes and iris points
            for (x, y) in [left_iris, right_iris]:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            
            # Display gaze direction
            cv2.putText(frame, gaze_horizontal, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, gaze_vertical, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow('Eye Gaze Estimation', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
