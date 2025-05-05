import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import socket

# # Bağlantılar (şu an için kapalı)
# PORT = 9999
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.connect(("192.168.68.66", PORT))

# MediaPipe Pose ve RealSense kurulumu
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)

# RealSense pipeline kurulumu
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

def draw_arrow(img, p1, p2, color, label):
    p1 = tuple(int(v) for v in p1)
    p2 = tuple(int(v) for v in p2)
    cv2.arrowedLine(img, p1, p2, color, 3, tipLength=0.2)
    cv2.putText(img, label, p2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def compute_plane_from_points(points):
    """
    4 noktadan (2 omuz, 2 kalça) düzlem hesaplar
    points: Sözlük formatında 3D noktalar
    """
    # Omuz merkezini hesapla
    shoulder_center = (points['LEFT_SHOULDER'] + points['RIGHT_SHOULDER']) / 2
    # Kalça merkezini hesapla
    hip_center = (points['LEFT_HIP'] + points['RIGHT_HIP']) / 2
    
    # Z ekseni: Omuz merkezinden kalça merkezine (gövde boyunca, aşağı)
    z_axis = hip_center - shoulder_center
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # Omuzlar arasındaki vektör
    shoulder_vec = points['RIGHT_SHOULDER'] - points['LEFT_SHOULDER']
    
    # X ekseni: Z ile omuz vektörünün çapraz çarpımı (vücut önü)
    x_axis = np.cross(z_axis, shoulder_vec)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Y ekseni: Z ve X eksenlerinin çapraz çarpımı (sağ taraf)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Vücut merkezi
    body_center = (shoulder_center + hip_center) / 2
    
    return x_axis, y_axis, z_axis, body_center

# Pencere ilk oluşturulduğunda bir kere çağrılır
cv2.namedWindow("Vücut Koordinat Sistemi", cv2.WINDOW_AUTOSIZE)

frame_counter = 0
print_interval = 1  # Her 10 karede bir ekrana yazma

try:
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
            
            # 4 nokta da tespit edildi mi?
            if len(points_3d) == 4:
                # Vücut düzlemini hesaplama
                x_axis, y_axis, z_axis, origin = compute_plane_from_points(points_3d)
                
                # Koordinat eksenlerini görselleştirme
                scale = 0.2
                x_end = origin + scale * x_axis
                y_end = origin + scale * y_axis
                z_end = origin + scale * z_axis
                
                origin_2d = world_to_pixel(intrinsics, origin.tolist())
                x_2d = world_to_pixel(intrinsics, x_end.tolist())
                y_2d = world_to_pixel(intrinsics, y_end.tolist())
                z_2d = world_to_pixel(intrinsics, z_end.tolist())
                
                if None not in (origin_2d, x_2d, y_2d, z_2d):
                    draw_arrow(annotated_image, origin_2d, x_2d, (0, 0, 255), 'X (Ön)')
                    draw_arrow(annotated_image, origin_2d, y_2d, (0, 255, 0), 'Y (Sağ)')
                    draw_arrow(annotated_image, origin_2d, z_2d, (255, 0, 0), 'Z (Aşağı)')
                
                # Kol açılarını hesaplama
                if (hasattr(mp_pose.PoseLandmark, 'RIGHT_ELBOW') and 
                    hasattr(mp_pose.PoseLandmark, 'RIGHT_SHOULDER') and 
                    hasattr(mp_pose.PoseLandmark, 'RIGHT_WRIST') and
                    hasattr(mp_pose.PoseLandmark, 'LEFT_ELBOW') and 
                    hasattr(mp_pose.PoseLandmark, 'LEFT_SHOULDER') and 
                    hasattr(mp_pose.PoseLandmark, 'LEFT_WRIST')):
                    
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
                    if (r_elbow_3d and r_shoulder_3d and r_wrist_3d and 
                        l_elbow_3d and l_shoulder_3d and l_wrist_3d):
                        
                        # Convert to numpy arrays
                        r_elbow = np.array(r_elbow_3d)
                        r_shoulder = np.array(r_shoulder_3d)
                        r_wrist = np.array(r_wrist_3d)
                        l_elbow = np.array(l_elbow_3d)
                        l_shoulder = np.array(l_shoulder_3d)
                        l_wrist = np.array(l_wrist_3d)
                        
                        # Create transformation matrix (dönüşüm matrisi)
                        # Yeni koordinat sistemini oluştur (x, y, z eksenlerini kullanarak)
                        R = np.vstack([x_axis, y_axis, z_axis])
                        
                        # RIGHT ARM CALCULATIONS
                        # Upper arm vector (üst kol vektörü)
                        r_upper_arm = r_elbow - r_shoulder
                        # Vektörü vücut koordinat sistemine dönüştür
                        r_upper_arm_body = np.dot(R, r_upper_arm)
                        
                        # Forearm vector (ön kol vektörü)
                        r_forearm = r_wrist - r_elbow
                        r_forearm_body = np.dot(R, r_forearm)
                        
                        # Shoulder roll (omuz yatay açısı, y-z düzleminde)
                        r_proj_yz = np.array([0, r_upper_arm_body[1], r_upper_arm_body[2]])
                        r_roll_rad = np.arctan2(r_proj_yz[1], r_proj_yz[2])
                        r_shoulder_roll = np.degrees(r_roll_rad)
                        # r_shoulder_roll = np.clip(r_shoulder_roll, -170, -90)
                        # r_shoulder_roll = np.interp(r_shoulder_roll, [-170, -90], [18, -76])
                        
                        # Shoulder pitch (omuz dikey açısı, x-z düzleminde)
                        r_proj_xz = np.array([r_upper_arm_body[0], 0, r_upper_arm_body[2]])
                        r_pitch_rad = np.arctan2(r_proj_xz[0], r_proj_xz[2])
                        r_shoulder_pitch = np.degrees(r_pitch_rad)
                        # r_shoulder_pitch = (r_shoulder_pitch+360)%360
                        # r_shoulder_pitch = np.clip(r_shoulder_pitch, 95, 170)
                        # r_shoulder_pitch = np.interp(r_shoulder_pitch, [95, 170], [-119.5, 0])
                        
                        # Elbow roll calculation (dirsek açısı)
                        r_upper_arm_norm = r_upper_arm / np.linalg.norm(r_upper_arm)
                        r_forearm_norm = r_forearm / np.linalg.norm(r_forearm)
                        r_dot = np.dot(r_upper_arm_norm, r_forearm_norm)
                        r_angle_rad = np.arccos(np.clip(r_dot, -1.0, 1.0))
                        r_elbow_roll = np.degrees(r_angle_rad)
                        # r_elbow_roll = np.clip(r_elbow_roll, 60, 170)
                        # r_elbow_roll = np.interp(r_elbow_roll, [60, 170], [88.5, 2])
                        
                        # LEFT ARM CALCULATIONS
                        # Upper arm vector
                        l_upper_arm = l_elbow - l_shoulder
                        l_upper_arm_body = np.dot(R, l_upper_arm)
                        
                        # Forearm vector
                        l_forearm = l_wrist - l_elbow
                        l_forearm_body = np.dot(R, l_forearm)
                        
                        # Shoulder roll calculation (y-Z plane)
                        l_proj_yz = np.array([0, l_upper_arm_body[1], l_upper_arm_body[2]])
                        l_roll_rad = np.arctan2(l_proj_yz[1], l_proj_yz[2])
                        l_shoulder_roll = np.degrees(l_roll_rad)
                        # l_shoulder_roll = np.clip(l_shoulder_roll, 90, 170)
                        # l_shoulder_roll = np.interp(l_shoulder_roll, [90, 170], [76, -18])
                        
                        # Shoulder pitch calculation (X-Z plane)
                        l_proj_xz = np.array([l_upper_arm_body[0], 0, l_upper_arm_body[2]])
                        l_pitch_rad = np.arctan2(l_proj_xz[0], l_proj_xz[2])
                        l_shoulder_pitch = np.degrees(l_pitch_rad)
                        # l_shoulder_pitch = np.clip(l_shoulder_pitch, 95, 180)
                        # l_shoulder_pitch = np.interp(l_shoulder_pitch, [95, 180], [-119.5, 0])
                        
                        # Elbow roll calculation
                        l_upper_arm_norm = l_upper_arm / np.linalg.norm(l_upper_arm)
                        l_forearm_norm = l_forearm / np.linalg.norm(l_forearm)
                        l_dot = np.dot(l_upper_arm_norm, l_forearm_norm)
                        l_angle_rad = np.arccos(np.clip(l_dot, -1.0, 1.0))
                        l_elbow_roll = np.degrees(l_angle_rad)
                        # l_elbow_roll = np.clip(l_elbow_roll, 60, 170)
                        # l_elbow_roll = np.interp(l_elbow_roll, [60, 170], [-88.5, -2])
                        
                        # Her 10 karede bir ekrana yazdır
                        frame_counter += 1
                        if frame_counter % print_interval == 0:
                            # 3D Koordinat bilgilerini yazdır
                            print("\n=== 3D KOORDİNATLAR ===")
                            print(f"SAĞ OMUZ : ({r_shoulder[0]:.3f}, {r_shoulder[1]:.3f}, {r_shoulder[2]:.3f})")
                            print(f"SOL OMUZ : ({l_shoulder[0]:.3f}, {l_shoulder[1]:.3f}, {l_shoulder[2]:.3f})")
                            print(f"SAĞ DİRSEK: ({r_elbow[0]:.3f}, {r_elbow[1]:.3f}, {r_elbow[2]:.3f})")
                            print(f"SOL DİRSEK: ({l_elbow[0]:.3f}, {l_elbow[1]:.3f}, {l_elbow[2]:.3f})")
                            print(f"SAĞ BİLEK : ({r_wrist[0]:.3f}, {r_wrist[1]:.3f}, {r_wrist[2]:.3f})")
                            print(f"SOL BİLEK : ({l_wrist[0]:.3f}, {l_wrist[1]:.3f}, {l_wrist[2]:.3f})")
                            
                            # Açı değerlerini yazdır
                            print("\n=== AÇI DEĞERLERİ ===")
                            print(f"Sağ Omuz Roll : {r_shoulder_roll:.2f}°")
                            print(f"Sağ Omuz Pitch : {r_shoulder_pitch:.2f}°")
                            print(f"Sağ Dirsek Roll : {r_elbow_roll:.2f}°")
                            print(f"Sol Omuz Roll : {l_shoulder_roll:.2f}°")
                            print(f"Sol Omuz Pitch : {l_shoulder_pitch:.2f}°")
                            print(f"Sol Dirsek Roll : {l_elbow_roll:.2f}°")
                            print("=====================\n")
                            
                            # # Socket üzerinden veri gönderimi
                            # try:
                            #     message_to_send = f"({r_shoulder_roll:.2f}, {r_shoulder_pitch:.2f}, {r_elbow_roll:.2f}, {l_shoulder_roll:.2f}, {l_shoulder_pitch:.2f}, {l_elbow_roll:.2f})\n"
                            #     # Uncomment to send over socket
                            #     # sock.sendall(message_to_send.encode())
                            # except (BrokenPipeError, ConnectionResetError):
                            #     print("Connection closed.")
                            #     break
                        
                        # Kolların vücut koordinat sistemindeki görselleştirmesi
                        scale = 0.2
                        
                        # Right arm visualization
                        r_end = origin \
                            + x_axis * r_upper_arm_body[0] * scale \
                            + y_axis * r_upper_arm_body[1] * scale \
                            + z_axis * r_upper_arm_body[2] * scale
                        
                        r_origin_2d = world_to_pixel(intrinsics, r_shoulder.tolist())
                        r_end_2d = world_to_pixel(intrinsics, r_end.tolist())
                        
                        if r_origin_2d and r_end_2d:
                            draw_arrow(annotated_image, r_origin_2d, r_end_2d, (0, 255, 255), 'R_Arm')
                        
                        # Left arm visualization
                        l_end = origin \
                            + x_axis * l_upper_arm_body[0] * scale \
                            + y_axis * l_upper_arm_body[1] * scale \
                            + z_axis * l_upper_arm_body[2] * scale
                        
                        l_origin_2d = world_to_pixel(intrinsics, l_shoulder.tolist())
                        l_end_2d = world_to_pixel(intrinsics, l_end.tolist())
                        
                        if l_origin_2d and l_end_2d:
                            draw_arrow(annotated_image, l_origin_2d, l_end_2d, (255, 0, 255), 'L_Arm')
        
        # Tek bir pencere olarak göster
        cv2.imshow("Vücut Koordinat Sistemi", annotated_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program kullanıcı tarafından durduruldu.")
finally:
    # Kaynakları temizle
    pipeline.stop()
    cv2.destroyAllWindows()
    # if 'sock' in locals():
    #     sock.close()
    print("Program sonlandırıldı.")