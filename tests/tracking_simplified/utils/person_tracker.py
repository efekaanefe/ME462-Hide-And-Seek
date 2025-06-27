import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy.optimize import linear_sum_assignment

class PersonTracker:
    """Handles tracking of people between frames."""
    
    def __init__(self):
        """Initialize the tracker."""
        self.next_id = 1
        self.tracked_objects = {}  # {id: track_data}
        self.disappear_threshold = 10.0  # seconds
        self.iou_threshold = -0.1
        self.max_age = 30  # frames
        
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
        
    def _hungarian_match(self, tracks: Dict[int, Dict], detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match tracks to detections using Hungarian algorithm."""
        if not tracks or not detections:
            return [], list(tracks.keys()), list(range(len(detections)))
            
        # Create cost matrix
        cost_matrix = np.ones((len(tracks), len(detections))) * float('inf')
        track_ids = list(tracks.keys())
        
        for i, track_id in enumerate(track_ids):
            track = tracks[track_id]
            for j, det in enumerate(detections):
                iou = self._calculate_iou(track['bbox'], det['bbox'])
                cost_matrix[i, j] = -iou  # Negative because Hungarian minimizes cost
                
        # Apply Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches by IoU threshold
        matches = []
        unmatched_tracks = []
        unmatched_detections = []
        
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < -self.iou_threshold:
                matches.append((track_ids[row], col))
            else:
                unmatched_tracks.append(track_ids[row])
                unmatched_detections.append(col)
                
        # Add remaining unmatched tracks and detections
        unmatched_tracks.extend([track_ids[i] for i in range(len(tracks)) 
                               if i not in row_indices])
        unmatched_detections.extend([j for j in range(len(detections)) 
                                   if j not in col_indices])
                                   
        return matches, unmatched_tracks, unmatched_detections
        
    def update(self, detections: List[Dict[str, Any]], current_time: float) -> Dict[int, Dict[str, Any]]:
        """Update tracks with new detections.
        
        Args:
            detections: List of detections
            current_time: Current timestamp
            
        Returns:
            Dictionary of active tracks
        """
        # Match existing tracks with detections
        matches, unmatched_tracks, unmatched_detections = self._hungarian_match(
            self.tracked_objects, detections
        )
        
        # Update matched tracks
        for track_id, det_idx in matches:
            det = detections[det_idx]
            self.tracked_objects[track_id].update({
                'bbox': det['bbox'],
                'last_seen': current_time,
                'age': 0
            })
            
            # Update additional information if available
            if 'recognized_name' in det:
                self.tracked_objects[track_id]['name'] = det['recognized_name']
            if 'map_position' in det:
                self.tracked_objects[track_id]['map_position'] = det['map_position']
            if 'orientation' in det:
                self.tracked_objects[track_id]['orientation'] = det['orientation']
            if 'map_orientation' in det:
                self.tracked_objects[track_id]['map_orientation'] = det['map_orientation']
                
            # Add track_id to detection
            det['track_id'] = track_id
                
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            new_track = {
                'bbox': det['bbox'],
                'first_seen': current_time,
                'last_seen': current_time,
                'age': 0
            }
            
            # Add additional information if available
            if 'recognized_name' in det:
                new_track['name'] = det['recognized_name']
            if 'map_position' in det:
                new_track['map_position'] = det['map_position']
            if 'orientation' in det:
                new_track['orientation'] = det['orientation']
                
            self.tracked_objects[self.next_id] = new_track
            # Add track_id to detection
            det['track_id'] = self.next_id
            self.next_id += 1
            
        # Update unmatched tracks
        for track_id in unmatched_tracks:
            track = self.tracked_objects[track_id]
            track['age'] += 1
            
            # Remove old tracks
            if track['age'] > self.max_age or \
               current_time - track['last_seen'] > self.disappear_threshold:
                del self.tracked_objects[track_id]
                
        return self.tracked_objects
        
    def visualize(self, frame: np.ndarray, tracks: Dict[int, Dict[str, Any]]) -> np.ndarray:
        """Draw tracking information on the frame.
        
        Args:
            frame: Input image frame
            tracks: Dictionary of active tracks
            
        Returns:
            Frame with visualization
        """
        vis_frame = frame.copy()
        
        for track_id, track in tracks.items():
            bbox = track['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw track ID and name
            name = track.get('name', 'Unknown')
            label = f"ID: {track_id} ({name})"
            cv2.putText(vis_frame, label, (x1, y2-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw orientation if available
            if 'orientation' in track:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                angle = track['orientation']
                
                arrow_length = 50
                end_x = int(center_x + arrow_length * np.cos(angle))
                end_y = int(center_y + arrow_length * np.sin(angle))
                
                cv2.arrowedLine(vis_frame, (center_x, center_y),
                              (end_x, end_y), (0, 255, 0), 2)
        
        return vis_frame 