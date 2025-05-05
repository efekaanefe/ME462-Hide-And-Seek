import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import socket


# # connections
# PORT = 9999
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.connect(("192.168.68.66", PORT))

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

def get_depth_at_pixel(depth_frame, x, y):
    if 0 <= x < depth_frame.width and 0 <= y < depth_frame.height:
        return depth_frame.get_distance(int(x), int(y))
    return 0.0

def pixel_to_world(depth_frame, intrinsics, x, y):
    depth = get_depth_at_pixel(depth_frame, x, y)
    if depth == 0.0:
        return None
    return rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)

def world_to_pixel(intrinsics, point3d):
    return rs.rs2_project_point_to_pixel(intrinsics, point3d)

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    annotated_image = color_image.copy()
    results = pose.process(color_image)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        keypoints = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP']
        points_2d = {}

        for name in keypoints:
            lm = getattr(mp_pose.PoseLandmark, name)
            pt = landmarks[lm]
            points_2d[name] = (pt.x * 640, pt.y * 480)

        points_3d = {}
        for name, (x, y) in points_2d.items():
            p = pixel_to_world(depth_frame, intrinsics, x, y)
            if p:
                points_3d[name] = np.array(p)

        if len(points_3d) == 4:
            sl = points_3d['LEFT_SHOULDER']
            sr = points_3d['RIGHT_SHOULDER']
            hl = points_3d['LEFT_HIP']
            hr = points_3d['RIGHT_HIP']

            # Y ekseni: sol omuzdan sağ omuza
            y_axis = sl - sr
            y_axis /= np.linalg.norm(y_axis)

            # Plane vektörleri
            v1 = sr - sl
            v2 = hl - sl

            # X ekseni: Vücut önü (plane'e dik)
            x_axis = np.cross(v2,y_axis)
            x_axis /= np.linalg.norm(x_axis)

            # Z ekseni: Vücut düzlemine dik (dışa yön)
            z_axis = np.cross(x_axis, y_axis)
            z_axis /= np.linalg.norm(z_axis)
            origin = (sl + sr) / 2  # Shoulder center
            scale = 0.2

            x_end = origin + scale * x_axis
            y_end = origin + scale * y_axis
            z_end = origin + scale * z_axis

            origin_2d = world_to_pixel(intrinsics, origin.tolist())
            x_2d = world_to_pixel(intrinsics, x_end.tolist())
            y_2d = world_to_pixel(intrinsics, y_end.tolist())
            z_2d = world_to_pixel(intrinsics, z_end.tolist())

            def draw_arrow(img, p1, p2, color, label):
                p1 = tuple(int(v) for v in p1)
                p2 = tuple(int(v) for v in p2)
                cv2.arrowedLine(img, p1, p2, color, 3, tipLength=0.2)
                cv2.putText(img, label, p2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if None not in (origin_2d, x_2d, y_2d, z_2d):
                draw_arrow(annotated_image, origin_2d, x_2d, (0, 0, 255), 'X')
                draw_arrow(annotated_image, origin_2d, y_2d, (0, 255, 0), 'Y')
                draw_arrow(annotated_image, origin_2d, z_2d, (255, 0, 0), 'Z')

            # Sağ ve sol kol açılarını hesaplama - single if statement approach
            if (hasattr(mp_pose.PoseLandmark, 'RIGHT_ELBOW') and hasattr(mp_pose.PoseLandmark, 'RIGHT_SHOULDER') and hasattr(mp_pose.PoseLandmark, 'RIGHT_WRIST') and
                hasattr(mp_pose.PoseLandmark, 'LEFT_ELBOW') and hasattr(mp_pose.PoseLandmark, 'LEFT_SHOULDER'))  and hasattr(mp_pose.PoseLandmark, 'LEFT_WRIST') :
                
                # 2D landmarks for right arm
                r_elbow_2d = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                r_shoulder_2d = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                r_wrist_2d = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                
                # 2D landmarks for left arm
                l_elbow_2d = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                l_shoulder_2d = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                l_wrist_2d = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                
                # Convert 2D to 3D points for right arm
                r_elbow_3d = pixel_to_world(depth_frame, intrinsics, r_elbow_2d.x * 640, r_elbow_2d.y * 480)
                r_shoulder_3d = pixel_to_world(depth_frame, intrinsics, r_shoulder_2d.x * 640, r_shoulder_2d.y * 480)
                r_wrist_3d = pixel_to_world(depth_frame, intrinsics, r_wrist_2d.x * 640, r_wrist_2d.y * 480)
                
                # Convert 2D to 3D points for left arm
                l_elbow_3d = pixel_to_world(depth_frame, intrinsics, l_elbow_2d.x * 640, l_elbow_2d.y * 480)
                l_shoulder_3d = pixel_to_world(depth_frame, intrinsics, l_shoulder_2d.x * 640, l_shoulder_2d.y * 480)
                l_wrist_3d = pixel_to_world(depth_frame, intrinsics, l_wrist_2d.x * 640, l_wrist_2d.y * 480)
                
                # Process if all needed 3D points are available
                if r_elbow_3d and r_shoulder_3d and r_wrist_3d and l_elbow_3d and l_shoulder_3d and l_wrist_3d:
                    # Convert to numpy arrays
                    r_elbow = np.array(r_elbow_3d)
                    r_shoulder = np.array(r_shoulder_3d)
                    r_wrist = np.array(r_wrist_3d)
                    
                    l_elbow = np.array(l_elbow_3d)
                    l_shoulder = np.array(l_shoulder_3d)
                    l_wrist = np.array(l_wrist_3d)
                    
                    # Create transformation matrix
                    R = np.vstack([x_axis, y_axis, z_axis])
                    
                    # RIGHT ARM CALCULATIONS
                    # Upper arm vector
                    r_upper_arm = r_elbow - r_shoulder
                    r_upper_arm_body = np.dot(R, r_upper_arm)
                    
                    # Shoulder roll calculation (y-Z plane)
                    r_proj1 = np.array([0, r_upper_arm_body[1], r_upper_arm_body[2]])
                    r_roll_rad = np.arctan2(r_proj1[1], r_proj1[2])
                    r_shoulder_roll = np.degrees(r_roll_rad)
                    r_shoulder_roll = np.clip(r_shoulder_roll, -170, -90) 
                    r_shoulder_roll = np.interp(r_shoulder_roll, [-170, -90], [18, -76])
                    
                    # Shoulder pitch calculation (X-Z plane)
                    r_proj2 = np.array([r_upper_arm_body[0], 0, r_upper_arm_body[2]])
                    r_pitch_rad = np.arctan2(r_proj2[0], r_proj2[2])
                    r_shoulder_pitch = np.degrees(r_pitch_rad)
                    r_shoulder_pitch = np.clip(r_shoulder_pitch, 95, 160) 
                    r_shoulder_pitch = np.interp(r_shoulder_pitch, [95, 180], [-119.5, 0])
                    
                    # Elbow roll calculation
                    r_upper_arm_vec = r_shoulder - r_elbow  # Omuz → Dirsek
                    r_forearm_vec = r_wrist - r_elbow       # Dirsek → Bilek
                    r_upper_arm_vec /= np.linalg.norm(r_upper_arm_vec)
                    r_forearm_vec /= np.linalg.norm(r_forearm_vec)
                    r_dot = np.dot(r_upper_arm_vec, r_forearm_vec)
                    r_angle_rad = np.arccos(np.clip(r_dot, -1.0, 1.0))
                    r_elbow_roll = np.degrees(r_angle_rad)
                    r_elbow_roll = np.clip(r_elbow_roll, 60, 170) 
                    r_elbow_roll = np.interp(r_elbow_roll, [60, 170], [88.5, 2])
                    
                    # LEFT ARM CALCULATIONS
                    # Upper arm vector
                    l_upper_arm = l_elbow - l_shoulder
                    l_upper_arm_body = np.dot(R, l_upper_arm)
                    
                    # Shoulder roll calculation (y-Z plane)
                    l_proj1 = np.array([0, l_upper_arm_body[1], l_upper_arm_body[2]])
                    l_roll_rad = np.arctan2(l_proj1[1], l_proj1[2])
                    l_shoulder_roll = np.degrees(l_roll_rad)
                    l_shoulder_roll = np.clip(l_shoulder_roll, 90, 170) 
                    l_shoulder_roll = np.interp(l_shoulder_roll, [90, 170], [76, -18])
                    
                    # Shoulder pitch calculation (X-Z plane)
                    l_proj2 = np.array([l_upper_arm_body[0], 0, l_upper_arm_body[2]])
                    l_pitch_rad = np.arctan2(l_proj2[0], l_proj2[2])
                    l_shoulder_pitch = np.degrees(l_pitch_rad)
                    l_shoulder_pitch = np.clip(l_shoulder_pitch, 95, 180) 
                    l_shoulder_pitch = np.interp(l_shoulder_pitch, [95, 180], [-119.5, 0])
                    
                    # Elbow roll calculation
                    l_upper_arm_vec = l_shoulder - l_elbow  # Omuz → Dirsek
                    l_forearm_vec = l_wrist - l_elbow       # Dirsek → Bilek
                    l_upper_arm_vec /= np.linalg.norm(l_upper_arm_vec)
                    l_forearm_vec /= np.linalg.norm(l_forearm_vec)
                    l_dot = np.dot(l_upper_arm_vec, l_forearm_vec)
                    l_angle_rad = np.arccos(np.clip(l_dot, -1.0, 1.0))
                    l_elbow_roll = np.degrees(l_angle_rad)
                    l_elbow_roll = np.clip(l_elbow_roll, 60, 170) 
                    l_elbow_roll = np.interp(l_elbow_roll, [60, 170], [-88.5, -2])
                    
                    message = (
                        f"Right Shoulder Roll  : {r_shoulder_roll:.2f}°\n"
                        f"Right Shoulder Pitch : {r_shoulder_pitch:.2f}°\n"
                        f"Right Elbow Roll     : {r_elbow_roll:.2f}°\n"
                        # f"Left Shoulder Roll   : {l_shoulder_roll:.2f}°\n"
                        # f"Left Shoulder Pitch  : {l_shoulder_pitch:.2f}°\n"
                        # f"Left Elbow Roll      : {l_elbow_roll:.2f}°\n"
                    )

                    print(message)

                    # # Try to send data over socket
                    # try:
                    #     message_to_send = f"({r_shoulder_roll:.2f}, {r_shoulder_pitch:.2f}, {r_elbow_roll:.2f}, {l_shoulder_roll:.2f}, {l_shoulder_pitch:.2f}, {l_elbow_roll:.2f})\n"
                    #     # Uncomment to send over socket
                    #     sock.sendall(message_to_send.encode())
                    # except (BrokenPipeError, ConnectionResetError):
                    #     print("Connection closed.")
                    #     break
                    
                    # Visualization (if needed)
                    scale = 0.2
                    
                    # Right arm visualization
                    r_end = origin \
                        + x_axis * r_upper_arm_body[0] * scale \
                        + y_axis * r_upper_arm_body[1] * scale \
                        + z_axis * r_upper_arm_body[2] * scale
                    
                    r_origin_2d = world_to_pixel(intrinsics, origin.tolist())
                    r_end_2d = world_to_pixel(intrinsics, r_end.tolist())
                    
                    if r_origin_2d and r_end_2d:
                        draw_arrow(annotated_image, r_origin_2d, r_end_2d, (0, 255, 255), 'R_UpperArm (Body)')
                    
                    # Left arm visualization
                    l_end = origin \
                        + x_axis * l_upper_arm_body[0] * scale \
                        + y_axis * l_upper_arm_body[1] * scale \
                        + z_axis * l_upper_arm_body[2] * scale
                    
                    l_origin_2d = world_to_pixel(intrinsics, origin.tolist())
                    l_end_2d = world_to_pixel(intrinsics, l_end.tolist())
                    
                    if l_origin_2d and l_end_2d:
                        draw_arrow(annotated_image, l_origin_2d, l_end_2d, (255, 0, 255), 'L_UpperArm (Body)')
    cv2.imshow("Koordinat Sistemi (Plane-Based)", annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
