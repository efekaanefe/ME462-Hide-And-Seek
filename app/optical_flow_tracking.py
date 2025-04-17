import cv2
import numpy as np

class OpticalFlowTracker:
    """Optical flow-based tracker for person tracking"""
    
    def __init__(self):
        # Optical flow parameters
        self.prev_frame = None
        self.prev_gray = None
        self.optical_flow_points = {}  # {track_id: [points to track]}
        self.max_optical_flow_points = 10  # Maximum number of points to track per object
        self.optical_flow_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
    def reset(self):
        """Reset the tracker"""
        self.prev_frame = None
        self.prev_gray = None
        self.optical_flow_points = {}
        
    def _generate_bbox_tracking_points(self, frame, bbox, max_points=10):
        """Generate points within a bounding box for optical flow tracking"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within frame boundaries
        height, width = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        
        # Skip if bbox is too small
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return []
        
        # Get ROI
        roi = frame[y1:y2, x1:x2]
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi
        
        # Find good features to track
        points = cv2.goodFeaturesToTrack(
            roi_gray,
            maxCorners=max_points,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        if points is None:
            return []
        
        # Convert points to global frame coordinates
        global_points = []
        for point in points:
            x, y = point.ravel()
            global_points.append([x + x1, y + y1])  # Adjust to global coordinates
            
        return np.array(global_points, dtype=np.float32).reshape(-1, 1, 2)
    
    def update_tracks(self, current_frame, tracks):
        """Track points using optical flow between frames"""
        # If no previous frame, just store this one and return original tracks
        if self.prev_frame is None:
            self.prev_frame = current_frame.copy()
            if len(current_frame.shape) == 3:
                self.prev_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            else:
                self.prev_gray = current_frame.copy()
            return tracks
        
        # Convert current frame to grayscale
        if len(current_frame.shape) == 3:
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        else:
            current_gray = current_frame
        
        # Process each track
        updated_tracks = {}
        
        for track_id, track_data in tracks.items():
            # Skip inactive tracks
            if not track_data.get('active', False):
                updated_tracks[track_id] = track_data.copy()
                continue
            
            bbox = track_data['bbox']
            
            # If no tracking points for this track, initialize them
            if track_id not in self.optical_flow_points or len(self.optical_flow_points.get(track_id, [])) < 4:
                points = self._generate_bbox_tracking_points(
                    self.prev_frame, 
                    bbox, 
                    self.max_optical_flow_points
                )
                
                if len(points) == 0:
                    updated_tracks[track_id] = track_data.copy()
                    continue
                    
                self.optical_flow_points[track_id] = points
                updated_tracks[track_id] = track_data.copy()
                continue
            
            # Calculate optical flow
            points = self.optical_flow_points[track_id]
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, 
                current_gray, 
                points, 
                None, 
                **self.optical_flow_params
            )
            
            # Keep only valid points
            good_new = new_points[status == 1]
            good_old = points[status == 1]
            
            # If too few points remain, regenerate tracking points next time
            if len(good_new) < 4:  # Need at least 4 points for good tracking
                self.optical_flow_points[track_id] = np.array([], dtype=np.float32).reshape(0, 1, 2)
                updated_tracks[track_id] = track_data.copy()
                continue
                
            # Calculate the shift
            dx_list = []
            dy_list = []
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                nx, ny = new.ravel()
                ox, oy = old.ravel()
                dx_list.append(nx - ox)
                dy_list.append(ny - oy)
            
            # Filter out outliers using median
            dx_median = np.median(dx_list)
            dy_median = np.median(dy_list)
            
            # Update bbox using median shift
            x1, y1, x2, y2 = bbox
            new_bbox = (
                int(x1 + dx_median),
                int(y1 + dy_median),
                int(x2 + dx_median),
                int(y2 + dy_median)
            )
            
            # Ensure bbox is within frame
            height, width = current_frame.shape[:2]
            new_x1 = max(0, min(width-1, new_bbox[0]))
            new_y1 = max(0, min(height-1, new_bbox[1]))
            new_x2 = max(new_x1+10, min(width, new_bbox[2]))
            new_y2 = max(new_y1+10, min(height, new_bbox[3]))
            
            # Create updated track data
            updated_track_data = track_data.copy()
            updated_track_data['bbox'] = (new_x1, new_y1, new_x2, new_y2)
            updated_track_data['optical_flow_updated'] = True
            
            # Update tracking points for next frame
            self.optical_flow_points[track_id] = good_new.reshape(-1, 1, 2)
            
            # Store updated track
            updated_tracks[track_id] = updated_track_data
        
        # Update previous frame and gray
        self.prev_frame = current_frame.copy()
        self.prev_gray = current_gray.copy()
        
        return updated_tracks
    
    def draw_flow(self, frame, color=(0, 255, 0)):
        """Draw optical flow points and vectors for visualization"""
        vis_frame = frame.copy()
        
        for track_id, points in self.optical_flow_points.items():
            # Draw points
            for point in points:
                x, y = point.ravel()
                cv2.circle(vis_frame, (int(x), int(y)), 2, color, -1)
        
        return vis_frame 