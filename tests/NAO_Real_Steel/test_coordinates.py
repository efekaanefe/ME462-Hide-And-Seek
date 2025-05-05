import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import socket
from tabulate import tabulate  # For nice formatting of the output

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)

# Initialize RealSense camera
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

# Dictionary to store the landmark names for printing
LANDMARK_NAMES = {
    mp_pose.PoseLandmark.LEFT_SHOULDER: "LEFT_SHOULDER",
    mp_pose.PoseLandmark.RIGHT_SHOULDER: "RIGHT_SHOULDER",
    mp_pose.PoseLandmark.LEFT_ELBOW: "LEFT_ELBOW",
    mp_pose.PoseLandmark.RIGHT_ELBOW: "RIGHT_ELBOW",
    mp_pose.PoseLandmark.LEFT_WRIST: "LEFT_WRIST",
    mp_pose.PoseLandmark.RIGHT_WRIST: "RIGHT_WRIST",

    mp_pose.PoseLandmark.LEFT_HIP: "LEFT_HIP",
    mp_pose.PoseLandmark.RIGHT_HIP: "RIGHT_HIP",
}

# For drawing keypoints
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

frame_counter = 0
print_interval = 1  # Print every 30 frames to avoid console spam

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

    # Draw pose landmarks on the image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        # Get 3D coordinates for all landmarks
        landmark_coords_3d = {}
        landmarks_table = []

        for idx, landmark in enumerate(landmarks):
            if idx in LANDMARK_NAMES:
                name = LANDMARK_NAMES[idx]
                x, y = landmark.x * 640, landmark.y * 480
                world_point = pixel_to_world(depth_frame, intrinsics, x, y)
                
                if world_point:
                    world_point = np.array(world_point)
                    landmark_coords_3d[name] = world_point
                    landmarks_table.append([
                        name,
                        f"{world_point[0]:.3f}",  # X
                        f"{world_point[1]:.3f}",  # Y
                        f"{world_point[2]:.3f}",  # Z (depth)
                        f"{landmark.visibility:.2f}"  # Visibility score
                    ])

        # Print 3D coordinates periodically to avoid console spam
        frame_counter += 1
        if frame_counter % print_interval == 0:
            print("\n=== 3D LANDMARK COORDINATES (in meters) ===")
            print(tabulate(
                landmarks_table,
                headers=["Landmark", "X (m)", "Y (m)", "Z (m)", "Visibility"],
                tablefmt="grid"
            ))
            print("\n")

        # Original code for calculating body coordinate system
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

            # Rest of the original code for arm angles calculation
            # ...

    cv2.imshow("Body Pose with 3D Landmarks", annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()