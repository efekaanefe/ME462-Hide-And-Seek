import cv2
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple

# TODO: 

class HomographyProjector:
    """Handles 2D mapping of camera coordinates to map coordinates using homography."""
    
    def __init__(self, homography_file: str = "homography_matrices.json"):
        """Initialize the homography projector.
        
        Args:
            homography_file: Path to homography matrices JSON file
        """
        self.homography_matrices = {}
        self.current_room = 0
        self.current_camera = 0
        self.map_image = None
        self.load_homography_matrices(homography_file)
        
    def load_homography_matrices(self, input_file: str) -> None:
        """Load homography matrices from JSON file.
        
        Args:
            input_file: Path to homography matrices JSON file
        """
        try:
            with open(input_file, 'r') as f:
                self.homography_matrices = json.load(f)
        except Exception as e:
            print(f"Error loading homography matrices: {e}")
            
    def select_room(self, room_index: int) -> None:
        """Select room to use for projection.
        
        Args:
            room_index: Index of room to select
        """
        self.current_room = room_index
        # Load map image for the room
        map_path = f"rooms_database/room{room_index}/2Dmap.png"
        self.map_image = cv2.imread(map_path)
        if self.map_image is not None:
            self.map_image = cv2.resize(self.map_image, (300, 300))
            
    def select_camera(self, cam_index: int) -> None:
        """Select camera to use for projection.
        
        Args:
            cam_index: Index of camera to select
        """
        self.current_camera = cam_index
        
    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Project detections to 2D map coordinates.
        
        Args:
            detections: List of detections with bbox information
            
        Returns:
            Updated detections with map coordinates
        """
        if not self.homography_matrices:
            return detections
            
        matrix_key = f"room{self.current_room}_cam{self.current_camera}"
        if matrix_key not in self.homography_matrices:
            return detections
            
        homography_matrix = np.array(self.homography_matrices[matrix_key]["matrix"])
  
        mapped_detections = []
        for det in detections:
            bbox = det['bbox']
            # Use bottom center as foot position
            foot_x = (bbox[0] + bbox[2]) / 2
            foot_y = bbox[3]
            
            # Project to map coordinates
            try:
                # Project foot position
                mapped_point = cv2.perspectiveTransform(
                    np.array([[[foot_x, foot_y]]], dtype=np.float32),
                    homography_matrix
                )[0][0]
                
                # If orientation is available, project orientation vector
                if 'orientation' in det:
                    # Create a point in the direction of orientation (100 pixels away)
                    angle = det['orientation']
                    dir_x = foot_x + 100 * np.cos(angle)
                    dir_y = foot_y + 100 * np.sin(angle)
                    
                    # Project both points
                    dir_point = cv2.perspectiveTransform(
                        np.array([[[dir_x, dir_y]]], dtype=np.float32),
                        homography_matrix
                    )[0][0]
                    
                    # Calculate orientation in map space
                    map_angle = np.arctan2(dir_point[1] - mapped_point[1], 
                                         dir_point[0] - mapped_point[0])
                    det['map_orientation'] = map_angle
                
                det['map_position'] = (mapped_point[0], mapped_point[1])
                mapped_detections.append(det)
            except Exception as e:
                print(f"Error mapping point: {e}")
                continue
                
        return mapped_detections
        
        
    def visualize(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw map positions on the frame.
        
        Args:
            frame: Input image frame
            detections: List of detections with map coordinates
            
        Returns:
            Frame with visualization
        """
        if self.map_image is None:
            return frame
            
        vis_frame = frame.copy()
        map_overlay = np.zeros_like(vis_frame)
        
        # Draw map
        map_copy = self.map_image.copy()
        
        # Draw detections on map
        for det in detections:
            if 'map_position' in det:
                map_x, map_y = det['map_position']
                # Scale map coordinates to map image size
                map_x = int(map_x * map_copy.shape[1] / 1000)
                map_y = int(map_y * map_copy.shape[0] / 1000)

                # Draw orientation arrow if available
                if 'map_orientation' in det:
                    orientation = det['map_orientation']
                    arrow_length = 15
                    dx = int(arrow_length * np.cos(orientation))
                    dy = int(arrow_length * np.sin(orientation))
                    cv2.arrowedLine(map_copy, (map_x, map_y),
                                  (map_x + dx, map_y + dy),
                                  (0, 0, 0), 4)
                
                # Draw person circle
                cv2.circle(map_copy, (map_x, map_y), 5, (0, 0, 255), -1)
                
                # Draw name if available
                name = det.get('name', 'Unknown')
                cv2.putText(map_copy, name, (map_x + 5, map_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Overlay map on frame
        map_overlay[10:10+map_copy.shape[0], 10:10+map_copy.shape[1]] = map_copy
        vis_frame = cv2.addWeighted(vis_frame, 1, map_overlay, 1, 0)
        
        return vis_frame 