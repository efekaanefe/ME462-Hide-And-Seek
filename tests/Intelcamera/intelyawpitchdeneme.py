import pyrealsense2 as rs
import cv2
import numpy as np
import mediapipe as mp
import time
import math

def angle_between(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))

def signed_angle_between(v1, v2, axis='z'):
    angle = angle_between(v1, v2)
    cross = np.cross(v1, v2)
    sign = 1
    if axis == 'z' and cross[2] < 0:
        sign = -1
    elif axis == 'x' and cross[0] < 0:
        sign = -1
    return angle * sign

def compute_pitch(v):  # Y-Z düzleminde, X sıfırlanır
    v_yz = np.array([0, v[1], v[2]])
    ref = np.array([0, -1, 0])
    return signed_angle_between(v_yz, ref, axis='x')

def compute_roll(v):   # X-Y düzleminde, Z sıfırlanır
    v_xy = np.array([v[0], v[1], 0])
    ref = np.array([0, -1, 0])
    return signed_angle_between(v_xy, ref, axis='z')

# --- RealSense setup ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

profile = pipeline.get_active_profile()
depth_stream = profile.get_stream(rs.stream.depth)
intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

# --- MediaPipe setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def get_3d_coords(landmarks, idx, w, h, depth_frame):
    lm = landmarks[idx]
    px = int(lm.x * w)
    py = int(lm.y * h)
    if 0 <= px < w and 0 <= py < h:
        depth = depth_frame.get_distance(px, py)
        x, y, z = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], depth)
        return np.array([x, y, z])
    else:
        return np.array([0.0, 0.0, 0.0])

# --- Main Loop ---
last_print_time = time.time()
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        h, w, _ = color_image.shape

        results = pose.process(rgb_image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            mp_drawing.draw_landmarks(color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 3D coordinates
            l_sh = get_3d_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, w, h, depth_frame)
            l_el = get_3d_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW, w, h, depth_frame)
            l_wr = get_3d_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST, w, h, depth_frame)

            r_sh = get_3d_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, w, h, depth_frame)
            r_el = get_3d_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW, w, h, depth_frame)
            r_wr = get_3d_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST, w, h, depth_frame)

            # Shoulder → Elbow vektör
            l_vec = l_el - l_sh
            r_vec = r_el - r_sh

            # Dirsek açısı
            l_elbow = angle_between(l_sh - l_el, l_wr - l_el)
            r_elbow = angle_between(r_sh - r_el, r_wr - r_el)

            # Açıları hesapla
            l_pitch = compute_pitch(l_vec)
            l_roll  = compute_roll(l_vec)
            r_pitch = compute_pitch(r_vec)
            r_roll  = compute_roll(r_vec)

            # Terminal çıktısı (5 saniyede bir)
            now = time.time()
            if now - last_print_time >= 5:
                print("\n--- İŞARETLİ PITCH / ROLL AÇILARI ---")
                print(f"Sol  Pitch : {round(l_pitch,2)}°")
                print(f"Sol  Roll  : {round(l_roll,2)}°")
                print(f"Sol  Dirsek: {round(l_elbow,2)}°")
                print(f"Sağ Pitch  : {round(r_pitch,2)}°")
                print(f"Sağ Roll   : {round(r_roll,2)}°")
                print(f"Sağ Dirsek : {round(r_elbow,2)}°")
                last_print_time = now

            # Görüntüye yaz
            cv2.putText(color_image, f"L-Pitch: {round(l_pitch,1)}°", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.putText(color_image, f"L-Roll : {round(l_roll,1)}°", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.putText(color_image, f"L-Elbow: {round(l_elbow,1)}°", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            cv2.putText(color_image, f"R-Pitch: {round(r_pitch,1)}°", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.putText(color_image, f"R-Roll : {round(r_roll,1)}°", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.putText(color_image, f"R-Elbow: {round(r_elbow,1)}°", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        cv2.imshow("RealSense: Signed Pitch & Roll", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

