import torch
import cv2
import numpy as np
import torch.nn.functional as F
import time


class DepthEstimator:
    def __init__(self, model_type="MiDaS_small"): 
        # Load MiDaS
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas.eval()

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas.to(self.device)

        # Use appropriate transform
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform  # Use dpt_transform for DPT models

    def predict(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)

        # Generate depth map
        start_time = time.time()

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],  # Resize to original size
                mode="bicubic",
                align_corners=False
            ).squeeze()

        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        print(f"Depth estimation took {elapsed_ms:.2f} ms")

        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
        return depth_map


# Initialize the estimator
depth_estimator = DepthEstimator("MiDaS_small")  # MiDaS_small DPT_Hybrid DPT_Large

# Read image
img = cv2.imread("test.jpeg")

# Predict and get the colorized depth map
depth_colormap = depth_estimator.predict(img)

# Save or display
cv2.imwrite("depth.png", depth_colormap)
cv2.imshow("Depth", depth_colormap)
cv2.waitKey(0)
cv2.destroyAllWindows()
