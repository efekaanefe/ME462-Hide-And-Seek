import open3d as o3d
import numpy as np
import cv2

# Load color and depth as NumPy arrays
color_np = cv2.imread("test.jpeg")  # shape: (H, W, 3)
depth_np = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED)  # shape: (H, W)

# Resize color image to match depth image size
depth_height, depth_width = depth_np.shape
color_np = cv2.resize(color_np, (depth_width, depth_height), interpolation=cv2.INTER_AREA)

# Convert BGR to RGB
color_np = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)

# Convert to Open3D images
color_raw = o3d.geometry.Image(color_np)
depth_raw = o3d.geometry.Image(depth_np)

# Get dimensions from NumPy shape
height, width, _ = color_np.shape

# Define camera intrinsics (you can tune fx, fy, cx, cy based on your scene)
fx = fy = 525.0
cx = width / 2.0
cy = height / 2.0
intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# Create RGBD image
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw,
    depth_scale=1.0,       # Adjust if depth is in different scale (e.g., 0–255 or 0–1)
    convert_rgb_to_intensity=False
)

# Create point cloud
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

# Flip the point cloud to align with Open3D's coordinate system
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])

# Visualize
o3d.visualization.draw_geometries([pcd])
