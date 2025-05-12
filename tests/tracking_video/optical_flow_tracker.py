import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
import time
from collections import deque
import colorsys
import os

class PersonTracker:
    def __init__(self, yolo_model="yolov8n.pt", conf_threshold=0.3, distance_threshold=50, trail_length=30):
        """
        Initialize the person tracker with YOLOv8 and optical flow
        
        Args:
            yolo_model: Path to YOLOv8 model
            conf_threshold: Confidence threshold for YOLOv8 detections
            distance_threshold: Maximum distance to consider for tracking association
            trail_length: Number of previous positions to keep for trajectory visualization
        """
        self.model = YOLO(yolo_model)
        self.conf_threshold = conf_threshold
        self.distance_threshold = distance_threshold
        self.next_id = 0
        self.tracks = {}  # format: {id: {"bbox": [x1, y1, x2, y2], "center": (x, y), "age": int, "feature_points": [...], "lost": int}}
        self.max_lost_frames = 30
        self.track_colors = {}
        self.prev_gray = None
        self.track_history = {}  # Store tracking data for output
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.trail_length = trail_length
        self.trajectories = {}  # format: {id: deque([(x, y), ...], maxlen=trail_length)}
        
        # Parameters for optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Bounding box stability parameters
        self.bbox_history = {}  # Track recent bounding boxes for smoothing
        self.bbox_history_size = 5  # Number of frames to keep for smoothing
        self.min_track_age = 3  # Minimum age for a track to be considered stable
        self.max_bbox_ratio_change = 0.3  # Maximum allowed change in aspect ratio
        self.max_bbox_area_change = 0.4  # Maximum allowed change in area
    
    def generate_color(self, track_id):
        """Generate a unique color for a track ID"""
        if track_id not in self.track_colors:
            # Generate hue between 0 and 1, with full saturation and value
            hue = (track_id * 0.1) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            self.track_colors[track_id] = tuple(int(c * 255) for c in rgb)
        return self.track_colors[track_id]
    
    def extract_bbox_corners(self, bbox):
        """Extract bbox corners as feature points instead of using goodFeaturesToTrack"""
        x1, y1, x2, y2 = [float(v) for v in bbox]
        
        # Define the 4 corners + center point
        corners = [
            [x1, y1],  # top-left
            [x2, y1],  # top-right
            [x2, y2],  # bottom-right
            [x1, y2],  # bottom-left
            [(x1 + x2) / 2, (y1 + y2) / 2]  # center
        ]
        
        # Add some additional points along the edges for better tracking
        mid_top = [x1 + (x2 - x1) / 2, y1]
        mid_bottom = [x1 + (x2 - x1) / 2, y2]
        mid_left = [x1, y1 + (y2 - y1) / 2]
        mid_right = [x2, y1 + (y2 - y1) / 2]
        
        corners.extend([mid_top, mid_bottom, mid_left, mid_right])
        
        # Convert to the format expected by optical flow
        return np.array([[corner] for corner in corners], dtype=np.float32)
    
    def calculate_bbox_metrics(self, bbox):
        """Calculate area and aspect ratio of a bounding box"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        return area, aspect_ratio
    
    def is_valid_bbox_change(self, track_id, new_bbox):
        """Check if the new bounding box is a valid change from previous boxes"""
        if track_id not in self.bbox_history or len(self.bbox_history[track_id]) == 0:
            return True
        
        # Get the last bbox from history
        last_bbox = self.bbox_history[track_id][-1]
        
        # Calculate metrics for both boxes
        new_area, new_ratio = self.calculate_bbox_metrics(new_bbox)
        old_area, old_ratio = self.calculate_bbox_metrics(last_bbox)
        
        # Check if area change is reasonable
        if old_area > 0:
            area_change = abs(new_area - old_area) / old_area
            if area_change > self.max_bbox_area_change:
                return False
        
        # Check if aspect ratio change is reasonable
        if old_ratio > 0:
            ratio_change = abs(new_ratio - old_ratio) / old_ratio
            if ratio_change > self.max_bbox_ratio_change:
                return False
        
        return True
    
    def smooth_bbox(self, track_id, new_bbox):
        """Apply smoothing to bounding box using weighted average of recent boxes"""
        if track_id not in self.bbox_history:
            self.bbox_history[track_id] = deque(maxlen=self.bbox_history_size)
            self.bbox_history[track_id].append(new_bbox)
            return new_bbox
        
        # Add new bbox to history
        self.bbox_history[track_id].append(new_bbox)
        
        # If we have enough history, apply smoothing
        if len(self.bbox_history[track_id]) >= 3:
            # Apply exponential smoothing with more weight to recent frames
            weights = np.exp(np.linspace(0, 1, len(self.bbox_history[track_id])))
            weights = weights / np.sum(weights)  # Normalize
            
            smoothed_bbox = np.zeros(4)
            for i, bbox in enumerate(self.bbox_history[track_id]):
                smoothed_bbox += np.array(bbox) * weights[i]
            
            return smoothed_bbox.tolist()
        
        return new_bbox
    
    def update_tracks_with_optical_flow(self, frame):
        """Update track positions using optical flow"""
        if self.prev_gray is None:
            return
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for track_id, track in list(self.tracks.items()):
            if len(track["feature_points"]) > 0:
                # Calculate optical flow
                new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, track["feature_points"], None, **self.lk_params
                )
                
                # Keep only valid points
                if new_points is not None:
                    valid_points = new_points[status == 1]
                    if len(valid_points) > 0:
                        track["feature_points"] = valid_points.reshape(-1, 1, 2)
                        
                        # Calculate new bounding box from points
                        x_coords = valid_points[:, 0]
                        y_coords = valid_points[:, 1]
                        x1 = np.min(x_coords)
                        y1 = np.min(y_coords)
                        x2 = np.max(x_coords)
                        y2 = np.max(y_coords)
                        
                        # Proposed new bbox
                        proposed_bbox = [x1, y1, x2, y2]
                        
                        # Only update if the bbox change is reasonable
                        if self.is_valid_bbox_change(track_id, proposed_bbox):
                            # Apply smoothing
                            smoothed_bbox = self.smooth_bbox(track_id, proposed_bbox)
                            
                            # Update bbox and center
                            track["bbox"] = smoothed_bbox
                            x1, y1, x2, y2 = smoothed_bbox
                            track["center"] = ((x1 + x2) / 2, (y1 + y2) / 2)
                            
                            # Flag as tracked by optical flow
                            track["optical_flow_updated"] = True
                            
                            # Update trajectory
                            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                            if track_id in self.trajectories:
                                self.trajectories[track_id].append((cx, cy))
                    else:
                        track["feature_points"] = np.array([])
        
        self.prev_gray = gray
    
    def process_frame(self, frame, frame_number):
        """Process a single frame"""
        # Run object detection
        results = self.model(frame, classes=0, conf=self.conf_threshold)  # class 0 is person in COCO
        detections = []
        
        # Process YOLOv8 results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].item()
                if confidence >= self.conf_threshold:
                    # Filter out obviously invalid boxes
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Skip boxes that are too small or have invalid dimensions
                    if width <= 5 or height <= 5 or width / height > 4 or height / width > 4:
                        continue
                    
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "center": center,
                        "confidence": confidence
                    })
        
        # Update existing tracks with optical flow
        self.update_tracks_with_optical_flow(frame)
        
        # Reset optical flow flag and increment age
        for track in self.tracks.values():
            track["optical_flow_updated"] = False
            track["age"] += 1
            track["lost"] += 1
        
        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        # First pass: match to tracks already updated by optical flow (already have momentum)
        for track_id, track in self.tracks.items():
            if track["optical_flow_updated"]:
                continue  # Already updated by optical flow
                
            for i, detection in enumerate(detections):
                if i in matched_detections:
                    continue
                    
                track_center = track["center"]
                det_center = detection["center"]
                distance = np.sqrt((track_center[0] - det_center[0])**2 + (track_center[1] - det_center[1])**2)
                
                if distance < self.distance_threshold:
                    # Check if the bounding box change is valid
                    if self.is_valid_bbox_change(track_id, detection["bbox"]):
                        # Apply smoothing to the new bbox
                        smoothed_bbox = self.smooth_bbox(track_id, detection["bbox"])
                        
                        # Update track with new detection
                        self.tracks[track_id]["bbox"] = smoothed_bbox
                        x1, y1, x2, y2 = smoothed_bbox
                        self.tracks[track_id]["center"] = ((x1 + x2) / 2, (y1 + y2) / 2)
                        self.tracks[track_id]["lost"] = 0
                        
                        # Update trajectory
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        if track_id in self.trajectories:
                            self.trajectories[track_id].append((cx, cy))
                        
                        # Update feature points with bbox corners
                        self.tracks[track_id]["feature_points"] = self.extract_bbox_corners(smoothed_bbox)
                        
                        matched_tracks.add(track_id)
                        matched_detections.add(i)
                        break
        
        # Only create new tracks for detections with high confidence
        new_track_conf_threshold = self.conf_threshold * 1.2  # Higher threshold for new tracks
        
        # Add new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detections and detection["confidence"] >= new_track_conf_threshold:
                # Use bbox corners as feature points
                feature_points = self.extract_bbox_corners(detection["bbox"])
                
                # Create new track
                track_id = self.next_id
                self.tracks[track_id] = {
                    "bbox": detection["bbox"],
                    "center": detection["center"],
                    "feature_points": feature_points,
                    "age": 0,
                    "lost": 0,
                    "optical_flow_updated": False
                }
                
                # Initialize bbox history
                self.bbox_history[track_id] = deque(maxlen=self.bbox_history_size)
                self.bbox_history[track_id].append(detection["bbox"])
                
                # Initialize trajectory
                cx, cy = int(detection["center"][0]), int(detection["center"][1])
                self.trajectories[track_id] = deque([(cx, cy)], maxlen=self.trail_length)
                
                # Initialize track history
                self.track_history[track_id] = {
                    "frames": [frame_number],
                    "boxes": [detection["bbox"]],
                    "centers": [detection["center"]]
                }
                
                self.next_id += 1
        
        # Update track history for existing tracks
        for track_id in self.tracks:
            if track_id in self.track_history:
                self.track_history[track_id]["frames"].append(frame_number)
                self.track_history[track_id]["boxes"].append(self.tracks[track_id]["bbox"])
                self.track_history[track_id]["centers"].append(self.tracks[track_id]["center"])
        
        # Remove lost tracks
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]["lost"] > self.max_lost_frames:
                # Don't remove tracks from trajectory visualization,
                # but they won't be considered for future matches
                del self.tracks[track_id]
        
        # Check if we should merge any trajectories
        self.check_for_trajectory_merges()
    
    def check_for_trajectory_merges(self):
        """Check if any trajectories should be merged (i.e., same person was assigned different IDs)"""
        # Only consider tracks with sufficient history
        stable_tracks = {
            id: track for id, track in self.tracks.items() 
            if track["age"] >= self.min_track_age
        }
        
        # For each pair of tracks, check if they might be the same person
        for id1, track1 in stable_tracks.items():
            for id2, track2 in stable_tracks.items():
                if id1 >= id2:  # Avoid comparing a track with itself or duplicate comparisons
                    continue
                
                # Calculate distance between track centers
                x1, y1 = track1["center"]
                x2, y2 = track2["center"]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                # If tracks are close and have similar sizes, they might be the same person
                if distance < self.distance_threshold:  # Stricter threshold for merging
                    # Calculate IOU between bounding boxes
                    box1 = track1["bbox"]
                    box2 = track2["bbox"]
                    
                    # Calculate intersection
                    x1_inter = max(box1[0], box2[0])
                    y1_inter = max(box1[1], box2[1])
                    x2_inter = min(box1[2], box2[2])
                    y2_inter = min(box1[3], box2[3])
                    
                    if x2_inter > x1_inter and y2_inter > y1_inter:
                        area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                        iou = area_inter / (area1 + area2 - area_inter)
                        
                        # If IOU is high, merge the tracks
                        if iou > 0.5:
                            # Merge the younger track into the older one
                            older_id = id1 if track1["age"] > track2["age"] else id2
                            younger_id = id2 if older_id == id1 else id1
                            
                            # Append the trajectory of the younger track to the older one
                            if younger_id in self.trajectories and older_id in self.trajectories:
                                for point in self.trajectories[younger_id]:
                                    self.trajectories[older_id].append(point)
                                
                                # Also update track history
                                if younger_id in self.track_history and older_id in self.track_history:
                                    for i, frame in enumerate(self.track_history[younger_id]["frames"]):
                                        self.track_history[older_id]["frames"].append(frame)
                                        self.track_history[older_id]["boxes"].append(self.track_history[younger_id]["boxes"][i])
                                        self.track_history[older_id]["centers"].append(self.track_history[younger_id]["centers"][i])
                                
                                # Keep the older track, remove the younger one
                                if younger_id in self.tracks:
                                    del self.tracks[younger_id]
                                # Don't delete the trajectory data for visualization
    
    def draw_tracks(self, frame):
        """Draw tracks and trajectories on the frame for visualization"""
        # Calculate FPS
        self.new_frame_time = time.time()
        fps = 1/(self.new_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = self.new_frame_time
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw tracks and trajectories
        for track_id, track in self.tracks.items():
            x1, y1, x2, y2 = [int(v) for v in track["bbox"]]
            color = self.generate_color(track_id)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw feature points
            for point in track["feature_points"]:
                x, y = point.ravel().astype(int)
                cv2.circle(frame, (x, y), 2, color, -1)
            
            # Draw trajectory (path)
            if track_id in self.trajectories and len(self.trajectories[track_id]) > 1:
                points = list(self.trajectories[track_id])
                # Draw lines connecting trajectory points
                for i in range(1, len(points)):
                    # Calculate alpha for fade effect (older points are more transparent)
                    alpha = 0.6 * (i / len(points))
                    thickness = 2
                    cv2.line(frame, points[i-1], points[i], color, thickness)
                
                # Draw trajectory points (small circles at each position)
                for i, point in enumerate(points):
                    # Smaller circles for older positions
                    size = max(1, int(2 * (i / len(points) + 0.5)))
                    cv2.circle(frame, point, size, color, -1)
        
        # Also draw trajectories for recently lost tracks (with fading effect)
        for track_id in list(self.trajectories.keys()):
            if track_id not in self.tracks and len(self.trajectories[track_id]) > 1:
                points = list(self.trajectories[track_id])
                color = self.generate_color(track_id)
                
                # Draw the trajectory with fading effect
                for i in range(1, len(points)):
                    # Older lines are more transparent
                    alpha = 0.3 * (i / len(points))
                    thickness = 1
                    cv2.line(frame, points[i-1], points[i], color, thickness)
        
        return frame

    def draw_heatmap(self, frame_shape, alpha=0.6):
        """Generate a heatmap of all trajectories"""
        heatmap = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
        
        # Accumulate all trajectory points
        for track_id, trajectory in self.trajectories.items():
            points = list(trajectory)
            for point in points:
                if 0 <= point[0] < frame_shape[1] and 0 <= point[1] < frame_shape[0]:
                    cv2.circle(heatmap, point, 10, 255, -1)
        
        # Apply gaussian blur to create heatmap effect
        heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
        
        # Create colored heatmap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Create mask where heatmap has values
        mask = heatmap > 0
        
        # Create output image
        output = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
        output[mask] = heatmap_colored[mask]
        
        return output, mask, alpha

    def process_video(self, input_path, output_path, show_video=True, save_tracks=True, save_heatmap=True):
        """
        Process a video file, tracking people throughout the frames
        
        Args:
            input_path: Path to input video file
            output_path: Path to output file
            show_video: Whether to show video during processing
            save_tracks: Whether to save tracking data as JSON
            save_heatmap: Whether to save heatmap image of trajectories
        """
        # Open video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer
        output_filename = Path(output_path).stem
        output_video_path = str(Path(output_path).with_suffix('.mp4'))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_number = 0
        self.track_history = {}
        self.trajectories = {}
        self.bbox_history = {}
        
        print(f"Processing video: {input_path}")
        print(f"Frames: {frame_count}, FPS: {fps}, Resolution: {width}x{height}")
        
        start_time = time.time()
        
        # Store the first frame for final visualization
        first_frame = None
        
        # Reset tracking state
        self.prev_gray = None
        self.next_id = 0
        self.tracks = {}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if first_frame is None:
                first_frame = frame.copy()
                
            # Process the frame
            self.process_frame(frame, frame_number)
            
            # Draw tracks on frame
            output_frame = self.draw_tracks(frame.copy())
            
            # Write frame to output video
            out.write(output_frame)
            
            # Display frame
            if show_video:
                cv2.imshow('Tracking', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_number += 1
            if frame_number % 100 == 0:
                elapsed = time.time() - start_time
                frames_per_second = frame_number / elapsed if elapsed > 0 else 0
                print(f"Processed {frame_number}/{frame_count} frames ({frames_per_second:.2f} FPS)")
        
        # Generate and save heatmap
        if save_heatmap and first_frame is not None:
            heatmap, mask, alpha = self.draw_heatmap((height, width))
            
            # Overlay heatmap on first frame
            heatmap_overlay = first_frame.copy()
            heatmap_overlay[mask] = cv2.addWeighted(first_frame, 1-alpha, heatmap, alpha, 0)[mask]
            
            # Save heatmap
            heatmap_path = str(Path(output_path).with_stem(f"{output_filename}_heatmap").with_suffix('.png'))
            cv2.imwrite(heatmap_path, heatmap_overlay)
            print(f"Heatmap saved to: {heatmap_path}")
        
        # Save tracking data as JSON
        if save_tracks:
            tracks_path = str(Path(output_path).with_suffix('.json'))
            with open(tracks_path, 'w') as f:
                # Convert trajectory deques to lists for JSON serialization
                trajectory_data = {}
                for track_id, trajectory in self.trajectories.items():
                    trajectory_data[track_id] = list(trajectory)
                
                json.dump({
                    "video_info": {
                        "path": input_path,
                        "width": width,
                        "height": height,
                        "fps": fps,
                        "frame_count": frame_count
                    },
                    "tracks": self.track_history,
                    "trajectories": trajectory_data
                }, f, indent=2)
            print(f"Tracking data saved to: {tracks_path}")
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Output video saved to: {output_video_path}")
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f}s ({frame_count/processing_time:.2f} FPS)")


def main():
    """Run the tracker without requiring command line arguments"""
    print("Person Tracking with YOLOv8 and Optical Flow")
    print("============================================")
    
    # Default settings
    input_path = "test-home.mp4"
    output_path = "test-home-out.mp4"
    yolo_model = "models/yolov8n.pt"
    conf_threshold = 0.5
    distance_threshold = 100
    trail_length = 30
    show_video = False
    save_heatmap = False
    save_tracks = False
    
    # Check if webcam should be used
    use_webcam = False
    
    # Initialize tracker
    tracker = PersonTracker(
        yolo_model=yolo_model,
        conf_threshold=conf_threshold,
        distance_threshold=distance_threshold,
        trail_length=trail_length
    )
    
    if use_webcam:
        # For webcam, we need a slightly different approach
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam")
            return
        
        # Get webcam properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30  # Assume 30 FPS for webcam
        
        # Setup video writer
        output_video_path = f"{output_path}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_number = 0
        start_time = time.time()
        first_frame = None
        
        print("Webcam tracking started. Press 'q' to stop.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if first_frame is None:
                first_frame = frame.copy()
            
            # Process frame
            tracker.process_frame(frame, frame_number)
            output_frame = tracker.draw_tracks(frame.copy())
            
            # Write and show frame
            out.write(output_frame)
            cv2.imshow('Tracking', output_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_number += 1
        
        # Save heatmap and tracking data
        if save_heatmap and first_frame is not None:
            heatmap, mask, alpha = tracker.draw_heatmap((height, width))
            heatmap_overlay = first_frame.copy()
            heatmap_overlay[mask] = cv2.addWeighted(first_frame, 1-alpha, heatmap, alpha, 0)[mask]
            heatmap_path = f"{output_path}_heatmap.png"
            cv2.imwrite(heatmap_path, heatmap_overlay)
            print(f"Heatmap saved to: {heatmap_path}")
        
        # Save tracking data
        tracks_path = f"{output_path}.json"
        with open(tracks_path, 'w') as f:
            trajectory_data = {}
            for track_id, trajectory in tracker.trajectories.items():
                trajectory_data[track_id] = list(trajectory)
            
            json.dump({
                "video_info": {
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "frame_count": frame_number
                },
                "tracks": tracker.track_history,
                "trajectories": trajectory_data
            }, f, indent=2)
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Output video saved to: {output_video_path}")
        print(f"Tracking data saved to: {tracks_path}")
        
    else:
        # Process video file
        tracker.process_video(
            input_path=input_path,
            output_path=output_path,
            show_video=show_video,
            save_heatmap=save_heatmap,
            save_tracks=save_tracks
        )


if __name__ == "__main__":
    main()