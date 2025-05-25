import torch
import cv2
import numpy as np
import torch.nn.functional as F

def create_custom_transform(target_height, target_width):
    def custom_transform(img):
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0), 
            size=(target_height, target_width), 
            mode='bilinear', 
            align_corners=False
        )
        return img_tensor
    return custom_transform

# Load MiDaS
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large") # DPT_Hybrid
midas.eval()

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# Read image
img = cv2.imread("test.jpeg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
original_height, original_width = img_rgb.shape[:2]

# Create custom transform for original dimensions  BADDDD
transform = create_custom_transform(original_height, original_width)

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform  # correct for DPT_Large

input_batch = transform(img_rgb).to(device)

# Predict depth
with torch.no_grad():
    prediction = midas(input_batch)

# Convert to numpy
depth = prediction.squeeze().cpu().numpy()

# Normalize for visualization
depth_min = depth.min()
depth_max = depth.max()
depth_vis = (255 * (depth - depth_min) / (depth_max - depth_min)).astype(np.uint8)

# Save depth map
cv2.imwrite("depth.png", depth_vis)