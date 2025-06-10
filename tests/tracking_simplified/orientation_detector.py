import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any, Optional, Tuple

class OrientationDetector:
    """Handles person orientation detection using MediaPipe."""
    
    def __init__(self):
        """Initialize the orientation detector."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        
        # Constants for direction weights
        self.DIRECTION_WEIGHT_NOSE = 0.2     # Weight given to nose direction
        self.DIRECTION_WEIGHT_SHOULDERS = 0.4 # Weight given to shoulder perpendicular
        self.DIRECTION_WEIGHT_FEET = 0.8      # Weight given to feet direction
        
        # Constants for landmark indices
        self.NOSE = self.mp_pose.PoseLandmark.NOSE
        self.LEFT_SHOULDER = self.mp_pose.PoseLandmark.LEFT_SHOULDER
        self.RIGHT_SHOULDER = self.mp_pose.PoseLandmark.RIGHT_SHOULDER
        self.LEFT_HIP = self.mp_pose.PoseLandmark.LEFT_HIP
        self.RIGHT_HIP = self.mp_pose.PoseLandmark.RIGHT_HIP
        self.LEFT_FOOT_INDEX = self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX
        self.RIGHT_FOOT_INDEX = self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
        self.LEFT_HEEL = self.mp_pose.PoseLandmark.LEFT_HEEL
        self.RIGHT_HEEL = self.mp_pose.PoseLandmark.RIGHT_HEEL
        
        # Visibility threshold for nose detection
        self.NOSE_VISIBILITY_THRESHOLD = 0.5
        
    def _get_landmark_coords(self, landmarks, landmark_idx):
        """Get 2D coordinates of a landmark."""
        landmark = landmarks[landmark_idx]
        return np.array([landmark.x, landmark.y])
        
    def update(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect orientation for each person.
        
        Args:
            frame: Input image frame
            detections: List of person detections
            
        Returns:
            Updated detections with orientation information
        """
        oriented_detections = []
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Extract person region
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                continue
                
            # Detect pose
            results = self.pose.process(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                key_points = {}
                
                # Convert landmarks to 2D coordinates
                for landmark_idx in [self.NOSE, self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
                                   self.LEFT_HIP, self.RIGHT_HIP, self.LEFT_FOOT_INDEX,
                                   self.RIGHT_FOOT_INDEX, self.LEFT_HEEL, self.RIGHT_HEEL]:
                    key_points[landmark_idx] = self._get_landmark_coords(landmarks, landmark_idx)
                
                # Calculate midpoints for reference
                shoulder_midpoint = None
                hip_midpoint = None
                
                if self.LEFT_SHOULDER in key_points and self.RIGHT_SHOULDER in key_points:
                    shoulder_midpoint = (key_points[self.LEFT_SHOULDER] + key_points[self.RIGHT_SHOULDER]) / 2
                
                if self.LEFT_HIP in key_points and self.RIGHT_HIP in key_points:
                    hip_midpoint = (key_points[self.LEFT_HIP] + key_points[self.RIGHT_HIP]) / 2
                
                # Initialize direction vectors and weights
                direction_vectors = []
                direction_weights = []
                
                # Vector 1: Shoulder line perpendicular
                if self.LEFT_SHOULDER in key_points and self.RIGHT_SHOULDER in key_points:
                    shoulder_vector = key_points[self.RIGHT_SHOULDER] - key_points[self.LEFT_SHOULDER]
                    perp_vector = np.array([-shoulder_vector[1], shoulder_vector[0]])
                    
                    norm = np.linalg.norm(perp_vector)
                    if norm > 0:
                        perp_vector = perp_vector / norm
                        
                        if hip_midpoint is not None and shoulder_midpoint is not None:
                            body_direction = hip_midpoint - shoulder_midpoint
                            x_projection = np.dot(perp_vector, [1, 0])
                            
                            if x_projection < 0:
                                perp_vector = -perp_vector
                                
                        direction_vectors.append(perp_vector)
                        direction_weights.append(self.DIRECTION_WEIGHT_SHOULDERS)
                
                # Vector 2: Nose direction
                if self.NOSE in key_points and shoulder_midpoint is not None and landmarks[self.NOSE].visibility > self.NOSE_VISIBILITY_THRESHOLD:
                    nose_vector = key_points[self.NOSE] - shoulder_midpoint
                    nose_vector = np.array([nose_vector[0], nose_vector[1]])
                    
                    norm = np.linalg.norm(nose_vector)
                    if norm > 0:
                        nose_vector = nose_vector / norm
                        
                        if key_points[self.NOSE][1] < shoulder_midpoint[1]:
                            direction_vectors.append(-nose_vector)
                        else:
                            direction_vectors.append(nose_vector)
                        direction_weights.append(self.DIRECTION_WEIGHT_NOSE)
                
                # Vector 3: Feet orientation
                feet_vector = None
                left_foot_vector = None
                right_foot_vector = None
                
                # Get left foot direction
                if self.LEFT_FOOT_INDEX in key_points and self.LEFT_HEEL in key_points:
                    left_foot_vector = key_points[self.LEFT_FOOT_INDEX] - key_points[self.LEFT_HEEL]
                    norm = np.linalg.norm(left_foot_vector)
                    if norm > 0:
                        left_foot_vector = left_foot_vector / norm
                
                # Get right foot direction
                if self.RIGHT_FOOT_INDEX in key_points and self.RIGHT_HEEL in key_points:
                    right_foot_vector = key_points[self.RIGHT_FOOT_INDEX] - key_points[self.RIGHT_HEEL]
                    norm = np.linalg.norm(right_foot_vector)
                    if norm > 0:
                        right_foot_vector = right_foot_vector / norm
                
                # Combine foot vectors
                if left_foot_vector is not None and right_foot_vector is not None:
                    feet_vector = (left_foot_vector + right_foot_vector) / 2
                    norm = np.linalg.norm(feet_vector)
                    if norm > 0:
                        feet_vector = feet_vector / norm
                        direction_vectors.append(feet_vector)
                        direction_weights.append(self.DIRECTION_WEIGHT_FEET)
                
                # Calculate final orientation
                if direction_vectors:
                    total_weight = sum(direction_weights)
                    if np.abs(total_weight) > 0:
                        norm_weights = [w / total_weight for w in direction_weights]
                        
                        front_direction = np.zeros(2)
                        for vector, weight in zip(direction_vectors, norm_weights):
                            front_direction += vector * weight
                        
                        norm = np.linalg.norm(front_direction)
                        if norm > 0:
                            front_direction = front_direction / norm
                            orientation = np.arctan2(front_direction[1], front_direction[0])
                            
                            # Store orientation information
                            det['orientation'] = orientation
                            det['landmarks'] = {
                                'shoulder_midpoint': shoulder_midpoint,
                                'hip_midpoint': hip_midpoint,
                                'nose': key_points.get(self.NOSE),
                                'left_shoulder': key_points.get(self.LEFT_SHOULDER),
                                'right_shoulder': key_points.get(self.RIGHT_SHOULDER)
                            }
                
            oriented_detections.append(det)
            
        return oriented_detections
        
    def visualize(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw orientation information on the frame.
        
        Args:
            frame: Input image frame
            detections: List of detections with orientation information
            
        Returns:
            Frame with visualization
        """
        vis_frame = frame.copy()
        
        for det in detections:
            if 'orientation' in det and 'landmarks' in det:
                bbox = det['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw orientation arrow
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                angle = det['orientation']
                
                # Calculate arrow end point
                arrow_length = 50
                end_x = int(center_x + arrow_length * np.cos(angle))
                end_y = int(center_y + arrow_length * np.sin(angle))
                
                # Draw arrow
                cv2.arrowedLine(vis_frame, (center_x, center_y),
                              (end_x, end_y), (0, 255, 0), 2)
                
                # Draw angle text
                angle_deg = np.degrees(angle)
                cv2.putText(vis_frame, f"{angle_deg:.1f}°",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)
        
        return vis_frame 