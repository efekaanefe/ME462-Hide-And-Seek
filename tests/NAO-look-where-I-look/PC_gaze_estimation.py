import cv2
import mediapipe as mp
import numpy as np
import socket

PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Socket connection is commented out, uncomment when needed
sock.connect(("192.168.68.66", PORT))

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # Needed to get iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

    img_h, img_w, _ = frame.shape

    face_2d = []
    face_3d = []


    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:  # Important points
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    
                    if idx == 1:  # Nose tip
                        nose_2d = (x, y)
                    
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = img_w
            cam_matrix = np.array([
                [focal_length, 0, img_w / 2],
                [0, focal_length, img_h / 2],
                [0, 0, 1]
            ])

            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix
            )

            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            # Convert to degrees
            x = angles[0] * 360  # Pitch
            y = angles[1] * 360  # Yaw
            z = angles[2] * 360  # Roll

            # Adjust yaw to be 0 when facing forward, positive to the right
            yaw = 2*y
            
            # Adjust pitch to be 0 when facing forward, positive when looking up
            pitch = -x  # Negate x because positive x is looking down in the original code

                        # Format for display
            yaw_display = yaw
            pitch_display = pitch

            # Try to send data over socket
            try:
                message = f"({yaw_display:.2f},{pitch_display:.2f})\n"
                # Uncomment to send over socket
                sock.sendall(message.encode())
            except (BrokenPipeError, ConnectionResetError):
                print("Connection closed.")
                break

            # Determine head orientation for visualization
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Looking Forward"

            # Draw the annotation on the image
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
            cv2.line(frame, p1, p2, (255, 0, 0), 3)
            
            # Display yaw and pitch values
            cv2.putText(frame, text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.2f} degrees", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.2f} degrees", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    cv2.imshow("Head Pose Estimation", frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
