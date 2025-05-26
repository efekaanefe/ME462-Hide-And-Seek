import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from homography_modular import HomographyTool
from typing import List, Dict, Tuple, Optional, Any
import mediapipe as mp
import torch
import torch.nn.functional as F
import time

YOLO_CONFIDENCE_THRESHOLD = 0.9 # TODO: create a config file 

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
        self.transform = midas_transforms.dpt_transform

    def predict(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        return depth_map

class PersonOrientationDetector:
    def __init__(self, homography_file: str = "homography_matrices.json", use_depth_orientation: bool = True):
        """
        Initialize the person orientation detector.
        
        Args:
            homography_file: Path to the homography matrices JSON file
            use_depth_orientation: Whether to use depth-based orientation estimation
        """
        self.homography_tool = HomographyTool()
        self.homography_tool.load_homography_matrices(homography_file)
        self.people_detector = None
        self.pose_detector = None
        self.mp_pose_landmarker = None
        self.yolo_model = None
        self.use_depth_orientation = use_depth_orientation
        self.depth_estimator = DepthEstimator("MiDaS_small") if use_depth_orientation else None
        self.current_frame = None
        self.initialize_models()
        
    def initialize_models(self) -> None:
        """Initialize person detection and pose estimation models"""
        # For better detection and orientation, use MediaPipe Pose
        try:
            self.mp = mp
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.mp_pose = mp.solutions.pose
            
            # Use mp.solutions.pose with min_detection_confidence set low enough to potentially detect multiple people
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.5,
                enable_segmentation=False
            )
            
            # For multi-person detection, we'll use MediaPipe's holistic model in conjunction with pose
            # But only if we need both face and body landmarks
            self.mp_holistic = mp.solutions.holistic
            self.holistic_detector = self.mp_holistic.Holistic(
                static_image_mode=True,
                model_complexity=1, 
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            print("Using MediaPipe for pose detection")
            
            # Try to load YOLOv8 for improved people detection
            try:
                # Check if ultralytics is installed
                import importlib
                ultralytics_spec = importlib.util.find_spec("ultralytics")
                if ultralytics_spec is not None:
                    from ultralytics import YOLO
                    
                    # Try to find YOLOv8 model weights
                    yolo_weights_paths = [
                        "models/yolov8n.pt",  # Nano model (smallest, fastest)
                        "models/yolov8s.pt",  # Small model
                        "yolov8n.pt",         # Check in current directory
                        os.path.join(os.path.expanduser("~"), "yolov8n.pt")  # Check in home directory
                    ]
                    
                    yolo_path = None
                    for path in yolo_weights_paths:
                        if os.path.exists(path):
                            yolo_path = path
                            break
                    
                    if yolo_path:
                        # Load YOLOv8 model
                        self.yolo_model = YOLO(yolo_path)
                        print(f"Using YOLOv8 model from {yolo_path}")
                    else:
                        # If weights not found, download the model
                        print("YOLOv8 weights not found, downloading YOLOv8n...")
                        self.yolo_model = YOLO("yolov8n.pt")
                        print("YOLOv8 model downloaded successfully")
                    
                else:
                    print("Ultralytics package not found. Installing...")
                    import subprocess
                    subprocess.check_call(['pip', 'install', 'ultralytics'])
                    
                    # Now try to import and load
                    from ultralytics import YOLO
                    self.yolo_model = YOLO("yolov8n.pt")  # This will download if not present
                    print("YOLOv8 model installed and loaded successfully")
                
            except Exception as e:
                print(f"Error setting up YOLOv8: {str(e)}")
                print("Falling back to HOG detector")
                # Fall back to HOG detector
                self.people_detector = cv2.HOGDescriptor()
                self.people_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
        except Exception as e:
            # Fallback to HOG detector if MediaPipe is not available
            print(f"MediaPipe error: {str(e)}")
            print("Using HOG detector as fallback")
            self.people_detector = cv2.HOGDescriptor()
            self.people_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    def detect_people(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect multiple people in the image and estimate their orientations.
        
        Args:
            image: Input image
            
        Returns:
            List of dictionaries containing detected people with their positions and orientations
        """
        people = []
        
        # First, detect people using YOLOv8 or HOG to get bounding boxes
        person_boxes = self._detect_person_boxes(image)
        
        if not person_boxes:
            print("No people detected by YOLOv8/HOG")
            # Try with MediaPipe directly on the full image as fallback
            people = self._detect_with_mediapipe(image)
            return people
        
        # For each detected person, crop the image and run pose estimation
        for i, (x, y, w, h, conf) in enumerate(person_boxes):
            # Add some margin to the bounding box
            margin = 0.2  # 20% margin
            x_margin = int(w * margin)
            y_margin = int(h * margin)
            
            # Calculate new coordinates with margin
            x1 = max(0, x - x_margin)
            y1 = max(0, y - y_margin)
            x2 = min(image.shape[1], x + w + x_margin)
            y2 = min(image.shape[0], y + h + y_margin)
            
            # Crop the image for the person
            person_img = image[y1:y2, x1:x2]
            
            if person_img.size == 0:
                continue  # Skip empty crops
                
            # Run MediaPipe pose detection on the cropped image
            mp_person = self._process_person_with_mediapipe(person_img, (x1, y1, x2-x1, y2-y1))
            
            if mp_person:
                # Add YOLOv8 detection confidence
                mp_person["yolo_confidence"] = conf
                people.append(mp_person)
            else:
                # If MediaPipe fails, use bounding box info with default orientation
                foot_x = x + w // 2
                foot_y = y + h  # Bottom of the bounding box
                
                people.append({
                    "position": (foot_x, foot_y),
                    "orientation": np.pi,  # Default: facing down
                    "confidence": conf,  # Use YOLOv8 confidence
                    "yolo_confidence": conf,
                    "bbox": (x, y, w, h)
                })
        
        return people
    
    def _detect_person_boxes(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect people in the image and return their bounding boxes.
        
        Args:
            image: Input image
            
        Returns:
            List of bounding boxes as (x, y, width, height, confidence)
        """
        boxes = []
        
        # Try YOLOv8 detection first if available
        if self.yolo_model is not None:
            try:
                # Run YOLOv8 detection
                results = self.yolo_model(image, classes=0)  # class 0 is person in COCO dataset
                
                # Process the results
                for result in results:
                    # Get the prediction boxes, convert tensor to numpy if needed
                    detections = result.boxes
                    
                    for i in range(len(detections)):
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, detections.xyxy[i][:4].tolist())
                        # Calculate width and height
                        w, h = x2 - x1, y2 - y1
                        # Get confidence score
                        conf = float(detections.conf[i])
                        
                        # Add detection if confidence is high enough
                        if conf > YOLO_CONFIDENCE_THRESHOLD:  # Adjust threshold as needed
                            boxes.append((x1, y1, w, h, conf))
                
                print(f"YOLOv8 detected {len(boxes)} people")
                
            except Exception as e:
                print(f"Error in YOLOv8 detection: {str(e)}")
                
        # If YOLOv8 didn't find any people or wasn't available, try HOG
        if not boxes and self.people_detector is not None:
            try:
                hog_boxes, weights = self.people_detector.detectMultiScale(
                    image, 
                    winStride=(8, 8),
                    padding=(4, 4), 
                    scale=1.05
                )
                
                for i, box in enumerate(hog_boxes):
                    x, y, w, h = box
                    conf = float(weights[i]) if i < len(weights) else 0.5
                    boxes.append((x, y, w, h, conf))
                    
                print(f"HOG detected {len(boxes)} people")
                
            except Exception as e:
                print(f"Error in HOG detection: {str(e)}")
                
        return boxes
    
    def _detect_with_mediapipe(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Use MediaPipe to detect people in the full image.
        
        Args:
            image: Input image
            
        Returns:
            List of detected people
        """
        people = []
        
        if self.pose_detector:
            # Convert image to RGB for MediaPipe
            mp_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 and image.shape[2] == 3 else image
            
            # Process the image with MediaPipe Pose
            results = self.pose_detector.process(mp_image)
            
            if results.pose_landmarks:
                # Process the detected person
                landmarks = results.pose_landmarks.landmark
                
                # Get image dimensions
                h, w = image.shape[:2]
                
                # Process landmarks to get orientation and position
                # Note: we pass the original landmarks object for visualization
                person_data = self._process_landmarks(landmarks, w, h, results.pose_landmarks, (0, 0), image)
                
                if person_data:
                    people.append(person_data)
        
        return people
    
    def _process_person_with_mediapipe(self, person_img: np.ndarray, full_bbox: Tuple[int, int, int, int]) -> Optional[Dict[str, Any]]:
        """
        Process a cropped person image with MediaPipe.
        
        Args:
            person_img: Cropped image of the person
            full_bbox: Bounding box in the original image (x, y, w, h)
            
        Returns:
            Dictionary with person data or None if detection fails
        """
        if not self.pose_detector:
            return None
            
        # Convert to RGB for MediaPipe
        mp_image = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB) if len(person_img.shape) == 3 and person_img.shape[2] == 3 else person_img
        
        # Process with MediaPipe
        results = self.pose_detector.process(mp_image)
        
        if not results.pose_landmarks:
            return None
            
        # Get image dimensions of the cropped image
        crop_h, crop_w = person_img.shape[:2]
        
        # Get the offset from the original image
        x_offset, y_offset = full_bbox[0], full_bbox[1]
        
        # Process landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Process landmarks to get orientation and position
        return self._process_landmarks(landmarks, crop_w, crop_h, results.pose_landmarks, (x_offset, y_offset), person_img)
    
    def _process_landmarks(self, landmarks, width: int, height: int, 
                           original_landmarks, offset: Tuple[int, int] = (0, 0),
                           current_image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Process pose landmarks to extract orientation and position using 3D coordinates.
        
        Args:
            landmarks: MediaPipe pose landmarks
            width: Image width
            height: Image height
            original_landmarks: Original landmark object for visualization
            offset: Offset (x, y) if landmarks are from a cropped image
            current_image: The current image being processed (for depth estimation)
            
        Returns:
            Dictionary with person data
        """
        # ===== TUNABLE PARAMETERS =====
        VISIBILITY_THRESHOLD = 0.5
        NOSE_VISIBILITY_THRESHOLD = 0.8
        FOOT_VISIBILITY_THRESHOLD = 0.4
        
        # ===== LANDMARK INDICES =====
        NOSE = 0
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        LEFT_FOOT_INDEX = 31
        RIGHT_FOOT_INDEX = 32
        LEFT_HEEL = 29
        RIGHT_HEEL = 30
        
        x_offset, y_offset = offset
        
        # ===== EXTRACT VISIBLE LANDMARKS =====
        key_points = {}
        key_points_3d = {}
        
        # Get depth map only if using depth-based orientation
        depth_map = None
        if self.use_depth_orientation:
            if current_image is not None:
                depth_map = self.depth_estimator.predict(current_image)
            else:
                depth_map = self.depth_estimator.predict(self.current_frame)
        
        for i, landmark in enumerate(landmarks):
            if landmark.visibility > VISIBILITY_THRESHOLD:
                # Convert normalized coordinates to image coordinates
                point = [
                    landmark.x * width + x_offset,
                    landmark.y * height + y_offset
                ]
                key_points[i] = np.array(point)
        
                # Get depth value only if using depth-based orientation
                if self.use_depth_orientation and depth_map is not None:
                    try:
                        depth = depth_map[int(point[1]), int(point[0])]
                        # Create 3D point (x, y, z) where z is depth
                        point_3d = np.array([point[0], point[1], depth])
                        key_points_3d[i] = point_3d
                    except IndexError:
                        # Handle out of bounds access
                        continue
        
        # ===== ORIENTATION CALCULATION =====
        # Default orientation (facing down)
        orientation = None
        
        # Constants for direction weights
        DIRECTION_WEIGHT_NOSE = 0.05    # Weight given to nose direction (0-1)
        DIRECTION_WEIGHT_SHOULDERS = 0.2 # Weight given to shoulder perpendicular (0-1)
        DIRECTION_WEIGHT_FEET = 0.7     # Weight given to feet direction (0-1)
        
        if self.use_depth_orientation:
            # Try to calculate orientation using shoulder-hip plane with 3D points
            if (LEFT_SHOULDER in key_points_3d and RIGHT_SHOULDER in key_points_3d and 
                LEFT_HIP in key_points_3d and RIGHT_HIP in key_points_3d):

                # Get 3D points
                sl = key_points_3d[LEFT_SHOULDER]
                sr = key_points_3d[RIGHT_SHOULDER]
                hl = key_points_3d[LEFT_HIP]
                hr = key_points_3d[RIGHT_HIP]
                
                # Y axis: from left shoulder to right shoulder
                y_axis = sl - sr
                y_axis /= np.linalg.norm(y_axis)
                
                # Plane vectors
                v1 = sr - sl
                v2 = hl - sl
                
                # X axis: Body front (perpendicular to plane)
                x_axis = np.cross(v2, y_axis)
                x_axis /= np.linalg.norm(x_axis)
                
                # Z axis: Perpendicular to body plane (outward direction)
                z_axis = np.cross(x_axis, y_axis)
                z_axis /= np.linalg.norm(z_axis)

                # Calculate 3D points for origin and direction
                origin = (sl + sr) / 2  # Shoulder center
                scale = 0.2
                z_end = origin + scale * z_axis

                # Camera intrinsics (these should be calibrated for your camera)
                fx = 1000  # focal length x
                fy = 1000  # focal length y
                cx = width / 2  # principal point x
                cy = height / 2  # principal point y

                # Convert 3D points to 2D using perspective projection
                def project_3d_to_2d(point_3d, depth):
                    # Perspective projection
                    x = (point_3d[0] * fx / depth) + cx
                    y = (point_3d[1] * fy / depth) + cy
                    return np.array([x, y])

                # Get depth values for origin and z_end points
                origin_depth = depth_map[int(origin[1]), int(origin[0])]
                z_end_depth = depth_map[int(z_end[1]), int(z_end[0])]

                # Project points to 2D
                origin_2d = project_3d_to_2d(origin, origin_depth)
                z_end_2d = project_3d_to_2d(z_end, z_end_depth)

                # Calculate direction vector in 2D
                direction_2d = origin_2d - z_end_2d
                direction_2d = direction_2d / np.linalg.norm(direction_2d)

                # Calculate orientation angle from 2D direction
                orientation = np.arctan2(direction_2d[1], direction_2d[0])

                # Debug visualization
                if current_image is not None:
                    # Draw points and direction
                    cv2.circle(current_image, (int(origin_2d[0]), int(origin_2d[1])), 5, (0, 255, 0), -1)
                    cv2.circle(current_image, (int(z_end_2d[0]), int(z_end_2d[1])), 5, (0, 0, 255), -1)
                    cv2.arrowedLine(current_image, 
                                  (int(origin_2d[0]), int(origin_2d[1])),
                                  (int(z_end_2d[0]), int(z_end_2d[1])),
                                  (255, 0, 0), 2)

        else:
            # Use 2D landmarks for weighted orientation estimation
            direction_vectors = []
            direction_weights = []
            
            # Calculate midpoints for reference
            shoulder_midpoint = None
            hip_midpoint = None
            
            if LEFT_SHOULDER in key_points and RIGHT_SHOULDER in key_points:
                shoulder_midpoint = (key_points[LEFT_SHOULDER] + key_points[RIGHT_SHOULDER]) / 2
            
            if LEFT_HIP in key_points and RIGHT_HIP in key_points:
                hip_midpoint = (key_points[LEFT_HIP] + key_points[RIGHT_HIP]) / 2
            
            # Vector 1: Shoulder line perpendicular (strongest indicator for side view)
            if LEFT_SHOULDER in key_points and RIGHT_SHOULDER in key_points:
                shoulder_vector = key_points[RIGHT_SHOULDER] - key_points[LEFT_SHOULDER]
                # Get perpendicular to shoulder line (90° rotation)
                perp_vector = np.array([-shoulder_vector[1], shoulder_vector[0]])
                
                # Normalize the vector
                norm = np.linalg.norm(perp_vector)
                if norm > 0:
                    perp_vector = perp_vector / norm
                    
                    # Determine if perpendicular should point forward or backward
                    if hip_midpoint is not None and shoulder_midpoint is not None:
                        body_direction = hip_midpoint - shoulder_midpoint
                        # Flip direction if needed based on body orientation
                        if body_direction[1] > 0 and np.dot(perp_vector, [body_direction[0], 0]) > 0:
                            perp_vector = -perp_vector
                            
                    direction_vectors.append(perp_vector)
                    direction_weights.append(DIRECTION_WEIGHT_SHOULDERS)
            
            # Vector 2: Nose direction (strongest indicator for front/back view)
            if NOSE in key_points and shoulder_midpoint is not None and landmarks[NOSE].visibility > NOSE_VISIBILITY_THRESHOLD:
                nose_vector = key_points[NOSE] - shoulder_midpoint
                # Project to horizontal plane
                nose_vector = np.array([nose_vector[0], 0])
                
                # Normalize the vector
                norm = np.linalg.norm(nose_vector)
                if norm > 0:
                    nose_vector = nose_vector / norm
                    direction_vectors.append(nose_vector)
                    direction_weights.append(DIRECTION_WEIGHT_NOSE)
            
            # Vector 3: Feet orientation (useful for determining walking direction)
            feet_vector = None
            
            # Try different combinations of foot landmarks for orientation
            left_foot_vector = None
            right_foot_vector = None
            
            # Get left foot direction (heel to tip)
            if LEFT_FOOT_INDEX in key_points and LEFT_HEEL in key_points:
                left_foot_vector = key_points[LEFT_FOOT_INDEX] - key_points[LEFT_HEEL]
                # Normalize
                norm = np.linalg.norm(left_foot_vector)
                if norm > 0:
                    left_foot_vector = left_foot_vector / norm
            
            # Get right foot direction (heel to tip)
            if RIGHT_FOOT_INDEX in key_points and RIGHT_HEEL in key_points:
                right_foot_vector = key_points[RIGHT_FOOT_INDEX] - key_points[RIGHT_HEEL]
                # Normalize
                norm = np.linalg.norm(right_foot_vector)
                if norm > 0:
                    right_foot_vector = right_foot_vector / norm
            
            # Find the best combination based on available data
            if left_foot_vector is not None and right_foot_vector is not None:
                # Average the two foot vectors for most accurate direction
                feet_vector = (left_foot_vector + right_foot_vector) / 2
                # Normalize the average
                norm = np.linalg.norm(feet_vector)
                if norm > 0:
                    feet_vector = feet_vector / norm
            
            if feet_vector is not None:
                # Emphasize horizontal component more than vertical
                feet_vector = np.array([feet_vector[0], feet_vector[1]])
                
                # Normalize the vector
                norm = np.linalg.norm(feet_vector)
                if norm > 0:
                    feet_vector = feet_vector / norm
                    direction_vectors.append(feet_vector)
                    direction_weights.append(DIRECTION_WEIGHT_FEET)
            
            # Blend available direction vectors using weights
            if direction_vectors:
                # Normalize weights
                total_weight = sum(direction_weights)
                if total_weight > 0:
                    norm_weights = [w / total_weight for w in direction_weights]
                    
                    # Calculate weighted average direction
                    front_direction = np.zeros(2)
                    for vector, weight in zip(direction_vectors, norm_weights):
                        front_direction += vector * weight
                    
                    # Normalize final direction
                    norm = np.linalg.norm(front_direction)
                    if norm > 0:
                        front_direction = front_direction / norm
                        orientation = np.arctan2(front_direction[1], front_direction[0])
        
        # ===== POSITION CALCULATION =====
        foot_position = None
        
        # Try to get foot position from foot landmarks
        if LEFT_FOOT_INDEX in key_points and RIGHT_FOOT_INDEX in key_points:
            foot_position = (key_points[LEFT_FOOT_INDEX] + key_points[RIGHT_FOOT_INDEX]) / 2
        elif LEFT_HEEL in key_points and RIGHT_HEEL in key_points:
            foot_position = (key_points[LEFT_HEEL] + key_points[RIGHT_HEEL]) / 2
        elif LEFT_ANKLE in key_points and RIGHT_ANKLE in key_points:
            foot_position = (key_points[LEFT_ANKLE] + key_points[RIGHT_ANKLE]) / 2
        
        if foot_position is None:
            # Try to use hip position as fallback
            if LEFT_HIP in key_points and RIGHT_HIP in key_points:
                hip_position = (key_points[LEFT_HIP] + key_points[RIGHT_HIP]) / 2
                # Estimate foot position below hip
                foot_position = np.array([
                    hip_position[0],
                    hip_position[1] + 100,  # Add some distance below hip
                    0  # No depth information when not using depth
                ])
            else:
                # Last resort: use the lowest visible point
                lowest_y = -1
                for point in key_points.values():
                    if point[1] > lowest_y:
                        lowest_y = point[1]
                        foot_position = np.array([point[0], point[1], 0])
        
        # If still no foot position, use center bottom of image
        if foot_position is None:
            foot_position = np.array([
                width / 2 + x_offset,
                height + y_offset,
                0  # No depth information when not using depth
            ])
        
        # ===== BOUNDING BOX CALCULATION =====
        if key_points:
            points_array = np.array(list(key_points.values()))
            min_x, min_y = np.min(points_array, axis=0)
            max_x, max_y = np.max(points_array, axis=0)
            bbox = (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
        else:
            # Estimate bbox from foot position
            foot_x, foot_y = foot_position[:2]
            estimated_height = 160
            estimated_width = estimated_height * 0.5
            bbox = (
                int(foot_x - estimated_width/2), 
                int(foot_y - estimated_height),
                int(estimated_width), 
                int(estimated_height)
            )
        
        # ===== CONFIDENCE CALCULATION =====
        confidence = 0.5
        
        # Boost confidence if we have key points
        num_key_points = len([p for p in [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP] 
                            if p in key_points])
        confidence = max(confidence, 0.3 + 0.1 * num_key_points)
        
        # ===== RETURN RESULT =====
        return {
            "position": (int(foot_position[0]), int(foot_position[1])),
            "orientation": orientation,
            "confidence": confidence,
            "bbox": bbox,
            "landmarks": landmarks,
            "original_landmarks": original_landmarks,
            "offset": offset,
            "dimensions": (width, height),
            "key_points_3d": key_points_3d if self.use_depth_orientation else None
        }
    
    def map_to_2d(self, people: List[Dict[str, Any]], room_index: int, cam_index: int) -> List[Dict[str, Any]]:
        """
        Map detected people from camera view to 2D map coordinates using homography.
        
        Args:
            people: List of detected people
            room_index: Room index
            cam_index: Camera index
            
        Returns:
            List of people with map coordinates
        """
        room = f"room{room_index}"
        cam = f"cam{cam_index}"
        key = f"{room}_{cam}"
        
        if key not in self.homography_tool.homography_matrices:
            raise ValueError(f"No homography matrix found for {key}")
        
        # Get homography matrix
        H = np.array(self.homography_tool.homography_matrices[key]["matrix"], dtype=np.float32)
        
        mapped_people = []
        for person in people:
            # Get foot position in camera coordinates
            cam_x, cam_y = person["position"]
            
            # Transform to map coordinates using homography
            map_point = cv2.perspectiveTransform(
                np.array([[[cam_x, cam_y]]], dtype=np.float32), 
                H
            )[0][0]
            
            map_x, map_y = map_point
            
            # Transform orientation
            # Basic transformation of angle using homography
            # This is an approximation - a more accurate approach would be to use
            # the Jacobian of the homography transformation at this point
            dx, dy = np.cos(person["orientation"]), np.sin(person["orientation"])
            cam_dir_point = (cam_x + dx * 20, cam_y + dy * 20)  # Point 20px in the direction of orientation
            
            # Transform the direction point to map coordinates
            map_dir_point = cv2.perspectiveTransform(
                np.array([[[cam_dir_point[0], cam_dir_point[1]]]], dtype=np.float32), 
                H
            )[0][0]
            
            # Calculate new orientation angle in map coordinates
            map_dx, map_dy = map_dir_point[0] - map_x, map_dir_point[1] - map_y
            map_orientation = np.arctan2(map_dy, map_dx)
            
            # Copy person data with mapped coordinates
            mapped_person = person.copy()
            mapped_person["map_position"] = (map_x, map_y)
            mapped_person["map_orientation"] = map_orientation
            
            mapped_people.append(mapped_person)
        
        return mapped_people
    
    def visualize_detection(self, image: np.ndarray, people: List[Dict[str, Any]], fps: float = 0.0) -> np.ndarray:
        """
        Visualize detected people and their orientations on the input image.
        
        Args:
            image: Input image
            people: List of detected people
            fps: Current FPS value
            
        Returns:
            Image with visualizations
        """
        # Create a copy of the image
        vis_image = image.copy()
        
        # Draw FPS
        cv2.putText(vis_image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # First draw bounding boxes and labels
        for i, person in enumerate(people):
            if person is None:
                continue
                
            # Generate unique color for this person
            color_r = (i * 65) % 256
            color_g = (i * 97) % 256
            color_b = (i * 111) % 256
            person_color = (color_b, color_g, color_r)  # BGR for OpenCV
            
            x, y, w, h = person["bbox"]
            center_x, center_y = person["position"]
            orientation = person["orientation"]
            confidence = person.get("confidence", 0.5)
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), person_color, 2)
            
            # Draw person ID
            cv2.putText(vis_image, f"Person {i+1}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, person_color, 2)
            
            # Draw confidence score
            cv2.putText(vis_image, f"Conf: {confidence:.2f}", (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, person_color, 2)
            
            # Draw foot position
            cv2.circle(vis_image, (center_x, center_y), 5, (0, 255, 255), -1)
        
        # Now draw pose skeleton for each person
        for i, person in enumerate(people):
            if person is None or 'landmarks' not in person:
                continue
                
            try:
                # Generate a unique color for this person's skeleton
                color_r = (i * 65) % 256
                color_g = (i * 97) % 256
                color_b = (i * 111) % 256
                pose_color = (color_b, color_g, color_r)  # BGR for OpenCV
                
                # Get the landmark information
                landmarks = person['landmarks']
                x_offset, y_offset = person.get('offset', (0, 0))
                orig_width, orig_height = person.get('dimensions', (0, 0))
                
                # Define the landmark connections for pose skeleton
                connections = [
                    # Torso
                    (11, 12),  # LEFT_SHOULDER, RIGHT_SHOULDER
                    (11, 23),  # LEFT_SHOULDER, LEFT_HIP
                    (12, 24),  # RIGHT_SHOULDER, RIGHT_HIP
                    (23, 24),  # LEFT_HIP, RIGHT_HIP
                    # Arms
                    (11, 13),  # LEFT_SHOULDER, LEFT_ELBOW
                    (12, 14),  # RIGHT_SHOULDER, RIGHT_ELBOW 
                    (13, 15),  # LEFT_ELBOW, LEFT_WRIST
                    (14, 16),  # RIGHT_ELBOW, RIGHT_WRIST
                    # Legs
                    (23, 25),  # LEFT_HIP, LEFT_KNEE
                    (24, 26),  # RIGHT_HIP, RIGHT_KNEE
                    (25, 27),  # LEFT_KNEE, LEFT_ANKLE
                    (26, 28),  # RIGHT_KNEE, RIGHT_ANKLE
                ]
                
                # Draw the connections
                for start_idx, end_idx in connections:
                    if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                        landmarks[start_idx].visibility > 0.5 and landmarks[end_idx].visibility > 0.5):
                        
                        start_x = int(landmarks[start_idx].x * orig_width + x_offset)
                        start_y = int(landmarks[start_idx].y * orig_height + y_offset)
                        end_x = int(landmarks[end_idx].x * orig_width + x_offset)
                        end_y = int(landmarks[end_idx].y * orig_height + y_offset)
                        
                        cv2.line(vis_image, (start_x, start_y), (end_x, end_y), pose_color, 2)
                
                # Draw key points
                for idx, landmark in enumerate(landmarks):
                    if landmark.visibility > 0.5:
                        point_x = int(landmark.x * orig_width + x_offset)
                        point_y = int(landmark.y * orig_height + y_offset)
                        cv2.circle(vis_image, (point_x, point_y), 4, pose_color, -1)
                        
            except Exception as e:
                print(f"Error drawing pose for person {i+1}: {str(e)}")
        
        return vis_image
    
    def _ensure_map_loaded(self, room_index: int, cam_index: int) -> bool:
        """
        Make sure the map image is loaded for the specified room and camera.
        
        Args:
            room_index: Room index
            cam_index: Camera index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to select the room and camera
            self.homography_tool.select_room(room_index)
            self.homography_tool.select_camera(cam_index)
            
            # Verify map_image is loaded and valid
            if self.homography_tool.map_image is None:
                print(f"Error: Map image not loaded for room{room_index}")
                return False
                
            # Check that map_image is a valid numpy array
            if not isinstance(self.homography_tool.map_image, np.ndarray):
                print(f"Error: Map image is not a valid numpy array, got {type(self.homography_tool.map_image)}")
                return False
                
            return True
        except Exception as e:
            print(f"Error loading map for room{room_index}: {str(e)}")
            return False
    
    def visualize_on_map(self, room_index: int, cam_index: int, mapped_people: List[Dict[str, Any]]) -> None:
        """
        Visualize detected people and their orientations on the 2D map.
        
        Args:
            room_index: Room index
            cam_index: Camera index
            mapped_people: List of mapped people
        """
        # Make sure map is loaded
        if not self._ensure_map_loaded(room_index, cam_index):
            print("Cannot visualize on map: Map not loaded properly")
            return
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Display the map image
        plt.imshow(self.homography_tool.map_image)
        plt.title(f"People Orientation Map - Room{room_index}, Cam{cam_index}")
        
        # Draw people as circles with orientation arrows
        for i, person in enumerate(mapped_people):
            if person is None:
                continue
                
            map_x, map_y = person["map_position"]
            orientation = person["map_orientation"]
            
            # Generate a unique color for this person
            color_r = (i * 65 % 256) / 255.0
            color_g = (i * 97 % 256) / 255.0
            color_b = (i * 111 % 256) / 255.0
            person_color = (color_r, color_g, color_b)
            
            # Draw circle
            circle = plt.Circle((map_x, map_y), 10, color=person_color, fill=True, alpha=0.7)
            plt.gca().add_patch(circle)
            
            # Draw person ID
            plt.text(map_x - 5, map_y - 20, f"Person {i+1}", color=person_color, fontsize=10, 
                    fontweight='bold', bbox=dict(facecolor='white', alpha=1.0))
            
            # Draw orientation arrow
            arrow_length = 20
            dx = arrow_length * np.cos(orientation)
            dy = arrow_length * np.sin(orientation)
            
            arrow = patches.Arrow(map_x, map_y, dx, dy, width=10, color='red')
            plt.gca().add_patch(arrow)
        
        plt.axis('on')
        plt.tight_layout()
        plt.show()
    
    def process_image(self, image_path: str, room_index: int, cam_index: int) -> None:
        """
        Process a single image to detect people, map them to 2D, and visualize.
        
        Args:
            image_path: Path to the input image
            room_index: Room index
            cam_index: Camera index
        """
        # First make sure the map image is properly loaded
        if not self._ensure_map_loaded(room_index, cam_index):
            print(f"Error: Could not load map for room{room_index}, cam{cam_index}")
            return
            
        # Load image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (1200, 1600))
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            print(f"Working directory: {os.getcwd()}")
            print(f"Checking if file exists: {os.path.exists(image_path)}")
            return
        
        # Store current frame for depth estimation
        self.current_frame = image
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Start timing
        start_time = time.time()
        
        # Detect people
        people = self.detect_people(image_rgb)
        print(f"Detected {len(people)} people in the image")
        
        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        
        if not people:
            print("No people detected")
            return
        
        # Map people to 2D
        try:
            mapped_people = self.map_to_2d(people, room_index, cam_index)
        except Exception as e:
            print(f"Error mapping people to 2D: {str(e)}")
            return
        
        # Visualize detections on the input image
        vis_image = self.visualize_detection(image_rgb, people, fps)
        
        # Display the results
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(vis_image)
        plt.title(f"Detected People: {len(people)}")
        plt.axis('on')
        
        plt.subplot(1, 2, 2)
        plt.imshow(self.homography_tool.map_image)
        plt.title(f"2D Map: {len(mapped_people)} People")
        
        # Draw people on the map
        for i, person in enumerate(mapped_people):
            if person is None:
                continue
                
            map_x, map_y = person["map_position"]
            orientation = person["map_orientation"]
            
            # Generate a unique color for this person
            color_r = (i * 65 % 256) / 255.0
            color_g = (i * 97 % 256) / 255.0
            color_b = (i * 111 % 256) / 255.0
            person_color = (color_r, color_g, color_b)
            
            # Draw circle
            circle = plt.Circle((map_x, map_y), 10, color=person_color, fill=True, alpha=0.7)
            plt.gca().add_patch(circle)
            
            # Draw person ID
            plt.text(map_x - 5, map_y - 20, f"Person {i+1}", color=person_color, fontsize=8)
            
            # Draw orientation arrow
            arrow_length = 20
            dx = arrow_length * np.cos(orientation)
            dy = arrow_length * np.sin(orientation)
            
            arrow = patches.Arrow(map_x, map_y, dx, dy, width=10, color='red')
            plt.gca().add_patch(arrow)
        
        plt.tight_layout()
        plt.show()
        
        # Also show a focused map view
        self.visualize_on_map(room_index, cam_index, mapped_people)


def main():
    """Main function to run the orientation detection"""
    # Path to homography matrices file
    homography_file = "homography_matrices.json"
    
    # Initialize the detector
    detector = PersonOrientationDetector(homography_file)
    
    # Get room and camera info
    room_index = 0
    cam_index = 0
    
    # Get input image
    image_path = "test_images/test2.png"
    # image_path = "test_images/test.jpeg"
    print(f"Using image: {image_path}")
    print(f"Room: {room_index}, Camera: {cam_index}")
    
    # Process the image
    detector.process_image(image_path, room_index, cam_index)


if __name__ == "__main__":
    main()
