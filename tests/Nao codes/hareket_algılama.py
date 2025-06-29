import cv2
import mediapipe as mp
import numpy as np
import socket

# NAO robotunun IP ve port bilgisi
NAO_IP = '192.168.0.202'
NAO_PORT = 5005

# MediaPipe kurulumu
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Açı hesaplama (2D düzlemde)
def calculate_2d_angle(p1, p2, origin):
    v1 = np.array([p1[0] - origin[0], p1[1] - origin[1]])
    v2 = np.array([p2[0] - origin[0], p2[1] - origin[1]])
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot = np.dot(unit_v1, unit_v2)
    angle_rad = np.arccos(np.clip(dot, -1.0, 1.0))
    return np.degrees(angle_rad)

# Mapleme (10–76 → 0–76 derece)
def map_angle(value, old_min=10, old_max=76, new_min=0, new_max=76):
    value = np.clip(value, old_min, old_max)
    return (value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

# Socket aç (client gibi davranacak)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((NAO_IP, NAO_PORT))

# Kamera başlat
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Pose işle
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        RS = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        RE = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        RH = mp_pose.PoseLandmark.RIGHT_HIP.value
        LS = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        LE = mp_pose.PoseLandmark.LEFT_ELBOW.value
        LH = mp_pose.PoseLandmark.LEFT_HIP.value

        r_shoulder = [lm[RS].x, lm[RS].y]
        r_elbow    = [lm[RE].x, lm[RE].y]
        r_hip      = [lm[RH].x, lm[RH].y]

        l_shoulder = [lm[LS].x, lm[LS].y]
        l_elbow    = [lm[LE].x, lm[LE].y]
        l_hip      = [lm[LH].x, lm[LH].y]

        # Açı hesapla
        r_roll = calculate_2d_angle(r_elbow, r_hip, r_shoulder)
        l_roll = calculate_2d_angle(l_elbow, l_hip, l_shoulder)

        # Map et
        r_mapped = map_angle(r_roll)
        l_mapped = map_angle(l_roll)

        # Veriyi gönder
        msg = "R:{:.1f},L:{:.1f}".format(r_mapped, l_mapped)
        client_socket.sendall(msg.encode())
        print("[GÖNDERİLDİ]", msg)

        # Ekrana iskelet çiz
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Shoulder Roll Real-time Send", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

client_socket.close()
cap.release()
cv2.destroyAllWindows()
pose.close()
