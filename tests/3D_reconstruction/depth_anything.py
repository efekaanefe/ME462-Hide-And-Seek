import torch
import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
import os
import time


def create_depth_map(rgb_image_path, output_path=None):
    """
    Generate depth map from RGB image using Depth Anything V2
    """
    
    # Load the image
    image = Image.open(rgb_image_path).convert('RGB')
    original_size = image.size  # (width, height)
    
    # Initialize Depth Anything V2 pipeline
    depth_estimator = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Generate depth map
    start_time = time.time()
    depth_map = depth_estimator(image)
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    print(f"Depth estimation took {elapsed_ms:.2f} ms")

    # Convert PIL depth map to numpy array
    depth_array = np.array(depth_map)
    
    # Resize depth map to match original image size
    depth_resized = cv2.resize(depth_array, original_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize depth values to 0-255 range for visualization
    depth_normalized = ((depth_resized - depth_resized.min()) / 
                       (depth_resized.max() - depth_resized.min()) * 255).astype(np.uint8)
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(rgb_image_path))[0]
        output_dir = os.path.dirname(rgb_image_path)
        output_path = os.path.join(output_dir, f"{base_name}_depth.png")
    
    # Save depth map
    cv2.imwrite(output_path, depth_normalized)
    
    return depth_resized


# Example usage
if __name__ == "__main__":
    # Single image processing
    rgb_path = "test.jpeg"
    depth_map = create_depth_map(rgb_path)
    