import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
import csv
import queue
from collections import deque
from scipy.spatial.distance import cosine # For feature comparison

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLOv8 model - using Ultralytics implementation
try:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')  # Load the smallest YOLOv8 model to start with
except ImportError:
    messagebox.showerror("Error", "Please install ultralytics: pip install ultralytics")
    model = None

# --- Add DeepFace import ---
try:
    from deepface import DeepFace
    # Optional: Preload models if needed, though DeepFace often handles this
    # print("Attempting to build face recognition model...")
    # DeepFace.build_model("Facenet") # Example: preload Facenet
    # print("Face recognition model built.")
except ImportError:
    messagebox.showerror("Error", "Please install deepface and mtcnn: pip install deepface mtcnn")
    DeepFace = None
except Exception as e:
    messagebox.showerror("Error", f"Error initializing DeepFace: {e}")
    DeepFace = None
# --- End DeepFace import ---

class PersonTracker:
    def __init__(self):
        self.next_id = 1
        self.tracked_objects = {}  # Active/inactive tracks
        self.face_database = {}  # Permanent storage: {id: {'feature': feature_vector, 'last_seen': timestamp}}
        self.disappear_threshold = 2.0
        # Don't remove tracks after reid_time_window - keep them forever in face_database
        self.reid_time_window = 1000.0  # Very long to effectively keep all tracks for re-id
        self.iou_threshold = 0.3
        self.feature_threshold = 0.4
        self.face_model_name = "Facenet"
        self.face_detector_backend = "mtcnn"
        
        # Add total_active_time to separate from elapsed time
        self.time_data = {}  # {id: {'first_seen': timestamp, 'last_seen': timestamp, 
                            #       'total_active_time': seconds, 'active_intervals': [(start, end), ...]}
        
        if DeepFace is None:
             print("WARNING: DeepFace library not available. Face re-identification will be disabled.")

    # +++ Helper: Extract Face Feature +++
    def _extract_face_feature(self, frame, bbox):
        """Extracts face embedding from the person bounding box."""
        if DeepFace is None: return None # Skip if library not loaded

        x1, y1, x2, y2 = map(int, bbox)
        # Ensure coordinates are valid
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        if x1 >= x2 or y1 >= y2:
            return None # Invalid bbox

        # Crop the person region from the frame
        person_img = frame[y1:y2, x1:x2]

        if person_img.size == 0:
            # print("Empty image crop for feature extraction.")
            return None

        try:
            # Use DeepFace.represent to detect the face and get embedding
            embedding_objs = DeepFace.represent(
                img_path=person_img,
                model_name=self.face_model_name,
                detector_backend=self.face_detector_backend,
                enforce_detection=True, # MUST find a face
                align=True # Align face improves accuracy
            )
            # represent returns a list of dicts, get embedding from the first face found
            if embedding_objs:
                return np.array(embedding_objs[0]['embedding'], dtype=np.float32)
            else:
                # print(f"No face detected by {self.face_detector_backend} within bbox {bbox}")
                return None
        except ValueError as e:
            # Handles cases like no face found when enforce_detection=True
            # print(f"Face representation error (ValueError) for bbox {bbox}: {e}")
            return None
        except Exception as e:
            # Catch other unexpected errors during extraction
            print(f"Unexpected error during face feature extraction for bbox {bbox}: {e}")
            return None

    # +++ Helper: Calculate Feature Distance +++
    def _calculate_feature_distance(self, feature1, feature2):
        """Calculates cosine distance between two feature vectors."""
        # scipy.spatial.distance.cosine calculates 1 - cosine_similarity
        try:
             # Ensure they are numpy arrays
             feature1 = np.asarray(feature1, dtype=np.float32)
             feature2 = np.asarray(feature2, dtype=np.float32)
             # Handle potential zero vectors (shouldn't happen with embeddings)
             if np.linalg.norm(feature1) == 0 or np.linalg.norm(feature2) == 0:
                 return 1.0 # Max distance
             dist = cosine(feature1, feature2)
             return dist if not np.isnan(dist) else 1.0 # Handle potential NaN
        except Exception as e:
             print(f"Error calculating feature distance: {e}")
             return 1.0 # Return max distance on error

    def update(self, frame, detections, current_time=None):
        """Update tracks using IoU and Face Features."""
        if current_time is None:
            current_time = time.time()
            
        active_objects = {}
        inactive_objects_for_reid = {}
        newly_detected_indices = set(range(len(detections)))
        matched_track_ids = set() # Tracks matched in this frame (IoU or Feature)

        # 1. Separate active and potentially re-identifiable inactive objects
        for obj_id, obj_data in self.tracked_objects.items():
            if obj_data.get('active', True):
                active_objects[obj_id] = obj_data
            elif current_time - obj_data['last_seen'] < self.reid_time_window:
                # Keep inactive objects for potential re-id
                inactive_objects_for_reid[obj_id] = obj_data

        # 2. Match detections to ACTIVE objects using IoU
        if active_objects and newly_detected_indices:
            active_track_ids = list(active_objects.keys())
            active_bboxes = [active_objects[tid]['bbox'] for tid in active_track_ids]
            detection_indices_list = list(newly_detected_indices)
            detection_bboxes = [detections[i]['bbox'] for i in detection_indices_list]

            if not detection_bboxes: # Skip if no detections left
                 pass
            else:
                iou_matrix = np.zeros((len(active_bboxes), len(detection_bboxes)))
                for i, track_bbox in enumerate(active_bboxes):
                    for j, det_bbox in enumerate(detection_bboxes):
                        iou_matrix[i, j] = self._calculate_iou(track_bbox, det_bbox)

                while iou_matrix.size > 0 and iou_matrix.max() > self.iou_threshold:
                    track_idx, det_list_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                    det_original_idx = detection_indices_list[det_list_idx]
                    track_id = active_track_ids[track_idx]

                    # IoU Match found for active track
                    bbox = detections[det_original_idx]['bbox']
                    self.tracked_objects[track_id]['bbox'] = bbox
                    self.tracked_objects[track_id]['last_seen'] = current_time
                    self.tracked_objects[track_id]['active'] = True # Ensure stays active
                    matched_track_ids.add(track_id)
                    newly_detected_indices.remove(det_original_idx) # Remove from pool of unmatched detections

                    # Update feature (optional, could be done less frequently)
                    # If feature is missing or very old, try updating
                    if self.tracked_objects[track_id].get('feature') is None:
                         new_feature = self._extract_face_feature(frame, bbox)
                         if new_feature is not None:
                             self.tracked_objects[track_id]['feature'] = new_feature

                    # Remove matched track and detection from further IoU matching this round
                    iou_matrix = np.delete(iou_matrix, track_idx, axis=0)
                    iou_matrix = np.delete(iou_matrix, det_list_idx, axis=1)
                    active_track_ids.pop(track_idx)
                    detection_indices_list.pop(det_list_idx)

                    if not detection_indices_list: break # No more detections left to match

        # 3. Match remaining detections to INACTIVE objects using IoU (for quick recovery)
        if inactive_objects_for_reid and newly_detected_indices:
            inactive_track_ids = list(inactive_objects_for_reid.keys())
            inactive_bboxes = [inactive_objects_for_reid[tid]['bbox'] for tid in inactive_track_ids]
            detection_indices_list = list(newly_detected_indices)
            detection_bboxes = [detections[i]['bbox'] for i in detection_indices_list]

            if not detection_bboxes:
                 pass
            else:
                iou_matrix = np.zeros((len(inactive_bboxes), len(detection_bboxes)))
                for i, track_bbox in enumerate(inactive_bboxes):
                    for j, det_bbox in enumerate(detection_bboxes):
                        iou_matrix[i, j] = self._calculate_iou(track_bbox, det_bbox)

                while iou_matrix.size > 0 and iou_matrix.max() > self.iou_threshold:
                    track_idx, det_list_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                    det_original_idx = detection_indices_list[det_list_idx]
                    track_id = inactive_track_ids[track_idx]

                    # IoU Match found for inactive track - Reactivate
                    print(f"Re-identified Person {track_id} by IoU")
                    bbox = detections[det_original_idx]['bbox']
                    self.tracked_objects[track_id]['bbox'] = bbox
                    self.tracked_objects[track_id]['last_seen'] = current_time
                    self.tracked_objects[track_id]['active'] = True # Reactivated!
                    matched_track_ids.add(track_id)
                    newly_detected_indices.remove(det_original_idx)

                    # Update feature if missing or on reactivation
                    if self.tracked_objects[track_id].get('feature') is None:
                         new_feature = self._extract_face_feature(frame, bbox)
                         if new_feature is not None:
                             self.tracked_objects[track_id]['feature'] = new_feature

                    # Remove matched track and detection
                    iou_matrix = np.delete(iou_matrix, track_idx, axis=0)
                    iou_matrix = np.delete(iou_matrix, det_list_idx, axis=1)
                    inactive_track_ids.pop(track_idx)
                    detection_indices_list.pop(det_list_idx)

                    if not detection_indices_list: break

        # 4. Try FEATURE-BASED Re-identification against ALL historical tracks, not just recent inactive ones
        detections_for_feature_check = list(newly_detected_indices)
        unmatched_detection_indices_final = set(newly_detected_indices)

        if DeepFace is not None and detections_for_feature_check:
            # Use ALL face database entries, not just inactive_objects_for_reid
            face_database_entries = {tid: data for tid, data in self.face_database.items() 
                                    if 'feature' in data and data['feature'] is not None}

            if face_database_entries:
                feature_matched_detections = set() # Indices of detections matched by feature

                for det_idx in detections_for_feature_check:
                    if det_idx in feature_matched_detections: continue # Already matched

                    bbox = detections[det_idx]['bbox']
                    det_feature = self._extract_face_feature(frame, bbox)

                    if det_feature is None:
                        continue # Cannot perform feature matching without a feature

                    best_match_id = -1
                    min_distance = self.feature_threshold # Use threshold as upper bound

                    # Compare det_feature with all available face database features
                    # Make a copy of keys to iterate over, as we might modify the dict
                    track_ids_to_check = list(face_database_entries.keys())
                    for track_id in track_ids_to_check:
                         # Check if track_id still available (might have been matched to another detection)
                         if track_id not in face_database_entries: continue

                         track_feature = face_database_entries[track_id]['feature']
                         distance = self._calculate_feature_distance(det_feature, track_feature)

                         if distance < min_distance:
                             min_distance = distance
                             best_match_id = track_id

                    # If a best match below threshold was found
                    if best_match_id != -1:
                         print(f"Re-identified Person {best_match_id} by FEATURE (Dist: {min_distance:.3f})")
                         
                         # Check if this person is already in tracked_objects
                         if best_match_id in self.tracked_objects:
                             # Reactivate existing track
                             self.tracked_objects[best_match_id]['active'] = True
                             self.tracked_objects[best_match_id]['bbox'] = bbox
                             self.tracked_objects[best_match_id]['last_seen'] = current_time
                         else:
                             # Recreate track with preserved ID
                             self.tracked_objects[best_match_id] = {
                        'bbox': bbox,
                                 'first_seen': current_time,  # Reset first seen to now
                        'last_seen': current_time,
                                 'active': True,
                                 'feature': det_feature
                    }
                    
                             # Add to time_data if needed
                             if best_match_id not in self.time_data:
                                 self.time_data[best_match_id] = {
                        'first_seen': current_time,
                        'last_seen': current_time,
                                     'total_active_time': 0,
                                     'active_intervals': []  # Track active time periods
                                 }
                             # Add new active interval
                             self.time_data[best_match_id]['active_intervals'].append([current_time, None])
                             
                         # Update database entry
                         self.face_database[best_match_id]['feature'] = det_feature  # Update the feature
                         self.face_database[best_match_id]['last_seen'] = current_time
                         
                         # Mark detection as matched by feature
                         feature_matched_detections.add(det_idx)
                         unmatched_detection_indices_final.remove(det_idx)

                         # Remove the matched track from pool for this frame's feature matching round
                         del face_database_entries[best_match_id]

        # 5. Create NEW tracks for truly unmatched detections
        for detection_idx in unmatched_detection_indices_final:
            bbox = detections[detection_idx]['bbox']
            new_feature = self._extract_face_feature(frame, bbox)

            new_id = self.next_id
            self.tracked_objects[new_id] = {
                'bbox': bbox,
                    'first_seen': current_time,
                    'last_seen': current_time,
                'active': True,
                'feature': new_feature
            }
            
            # Add to permanent face database if a feature was extracted
            if new_feature is not None:
                self.face_database[new_id] = {
                    'feature': new_feature,
                    'first_seen': current_time,
                    'last_seen': current_time
                }
            
            # Initialize time tracking with active intervals
            self.time_data[new_id] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                'total_active_time': 0,
                'active_intervals': [[current_time, None]]  # Start new active interval
                }
                
            matched_track_ids.add(new_id) # Add new ID to the set of 'currently present' tracks
            self.next_id += 1
            status = "with face feature" if new_feature is not None else "without face feature"
            print(f"New Person {new_id} detected ({status})")

        # 6. Update status of existing tracks that were NOT matched
        # When a person goes inactive, end their current active interval
        current_tracked_ids = set(self.tracked_objects.keys())
        unmatched_track_ids = current_tracked_ids - matched_track_ids # IDs not seen this frame

        for obj_id in unmatched_track_ids:
            if self.tracked_objects[obj_id].get('active', True):
                if current_time - self.tracked_objects[obj_id]['last_seen'] > self.disappear_threshold:
                    self.tracked_objects[obj_id]['active'] = False
                    
                    # Close the active interval and calculate duration
                    if obj_id in self.time_data:
                        intervals = self.time_data[obj_id]['active_intervals']
                        if intervals and intervals[-1][1] is None:
                            intervals[-1][1] = current_time  # Close the interval
                            interval_duration = current_time - intervals[-1][0]
                            self.time_data[obj_id]['total_active_time'] += interval_duration
                            self.time_data[obj_id]['last_seen'] = current_time
                    
                    print(f"Person {obj_id} marked inactive after {self.disappear_threshold:.1f} seconds")

        # 7. Clean up very old inactive tracks from tracked_objects (but keep in face_database)
        ids_to_remove = []
        for obj_id, obj_data in list(self.tracked_objects.items()):
             # Remove if inactive AND last seen time exceeds the re-id window
             if not obj_data.get('active', False) and (current_time - obj_data['last_seen'] > self.reid_time_window):
                 ids_to_remove.append(obj_id)
                 # Finalize time data if needed (already done when marked inactive)
                 print(f"Removing inactive Person {obj_id} from active tracking after {self.reid_time_window:.1f} seconds.")

        for obj_id in ids_to_remove:
            if obj_id in self.tracked_objects: # Check existence before deleting
                 del self.tracked_objects[obj_id]
            # Keep time_data entry for historical records.

        # 8. Update time data for ACTIVE objects only
        for obj_id, obj_data in self.tracked_objects.items():
            if obj_data.get('active', True):
                if obj_id in self.time_data:
                    # Update last_seen for active objects
                    self.time_data[obj_id]['last_seen'] = current_time
                    
                    # Make sure there's an open active interval
                    intervals = self.time_data[obj_id]['active_intervals']
                    if not intervals or intervals[-1][1] is not None:
                        # Start a new active interval if needed
                        intervals.append([current_time, None])
        
        return self.tracked_objects
    
    def get_time_data(self):
        """Get time data for all tracked objects"""
        current_time = time.time()
        result = []
        
        for obj_id, time_info in self.time_data.items():
            # Calculate total active time from intervals
            total_active_time = 0
            for interval in time_info['active_intervals']:
                start = interval[0]
                end = interval[1] if interval[1] is not None else (
                    current_time if obj_id in self.tracked_objects and 
                    self.tracked_objects[obj_id].get('active', False) else start
                )
                total_active_time += end - start
            
            # Determine if object is currently active
            active = obj_id in self.tracked_objects and self.tracked_objects[obj_id].get('active', False)
            
            result.append({
                'id': obj_id,
                'first_seen': datetime.fromtimestamp(time_info['first_seen']).strftime("%Y-%m-%d %H:%M:%S"),
                'last_seen': datetime.fromtimestamp(time_info['last_seen']).strftime("%Y-%m-%d %H:%M:%S"),
                'duration': self._format_duration(total_active_time),
                'duration_seconds': total_active_time,
                'status': 'Active' if active else 'Inactive'
            })
        
        return result
    
    def _format_duration(self, seconds):
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _calculate_iou(self, bbox1, bbox2):
        # Calculate intersection over union of two bounding boxes
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate area of intersection rectangle
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate area of both bounding boxes
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate union area
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        return intersection_area / union_area if union_area > 0 else 0


class CameraManager:
    def __init__(self, camera_ids=None):
        self.cameras = {}  # {id: {'cap': VideoCapture, 'frame': frame, 'thread': thread, 'running': bool}}
        self.frame_queues = {}  # {id: queue}
        self.frame_buffers = {}  # {id: deque} - to smooth out frame delivery
        
        if camera_ids:
            for cam_id in camera_ids:
                self.add_camera(cam_id)
    
    def add_camera(self, camera_id):
        """Add a new camera to the manager"""
        try:
            # Convert to int if it's a numeric string
            if isinstance(camera_id, str) and camera_id.isdigit():
                camera_id = int(camera_id)
                
            # Create video capture
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                return False, f"Could not open camera {camera_id}"
            
            # Create frame queue and buffer
            frame_queue = queue.Queue(maxsize=30)  # Buffer up to 30 frames
            frame_buffer = deque(maxlen=5)  # Smooth delivery with 5-frame buffer
            
            # Store camera info
            self.cameras[camera_id] = {
                'cap': cap,
                'frame': None,
                'thread': None,
                'running': False
            }
            self.frame_queues[camera_id] = frame_queue
            self.frame_buffers[camera_id] = frame_buffer
            
            return True, f"Camera {camera_id} added successfully"
        except Exception as e:
            return False, f"Error adding camera {camera_id}: {str(e)}"
    
    def start_camera(self, camera_id):
        """Start capturing frames from a camera"""
        if camera_id not in self.cameras:
            return False, f"Camera {camera_id} not found"
        
        if self.cameras[camera_id]['running']:
            return True, f"Camera {camera_id} already running"
        
        # Set running flag
        self.cameras[camera_id]['running'] = True
        
        # Start capture thread
        thread = threading.Thread(
            target=self._capture_frames,
            args=(camera_id,),
            daemon=True
        )
        thread.start()
        
        # Store thread
        self.cameras[camera_id]['thread'] = thread
        
        return True, f"Camera {camera_id} started"
    
    def stop_camera(self, camera_id):
        """Stop capturing frames from a camera"""
        if camera_id not in self.cameras:
            return False, f"Camera {camera_id} not found"
        
        if not self.cameras[camera_id]['running']:
            return True, f"Camera {camera_id} already stopped"
        
        # Clear the stop flag
        self.cameras[camera_id]['running'] = False
        
        # Wait for thread to finish
        if self.cameras[camera_id]['thread']:
            self.cameras[camera_id]['thread'].join(timeout=1.0)
        
        # Clear frame queue
        while not self.frame_queues[camera_id].empty():
            try:
                self.frame_queues[camera_id].get_nowait()
            except queue.Empty:
                break
        
        return True, f"Camera {camera_id} stopped"
    
    def get_frame(self, camera_id):
        """Get the latest frame from a camera"""
        if camera_id not in self.cameras:
            return False, None, f"Camera {camera_id} not found"
        
        # If buffer has frames, return the latest
        if self.frame_buffers[camera_id]:
            return True, self.frame_buffers[camera_id][-1], None
        
        # Otherwise return the last captured frame or None
        return True, self.cameras[camera_id]['frame'], None
    
    def start_all_cameras(self):
        """Start all cameras"""
        results = []
        for camera_id in self.cameras:
            success, message = self.start_camera(camera_id)
            results.append((camera_id, success, message))
        return results
    
    def stop_all_cameras(self):
        """Stop all cameras"""
        results = []
        for camera_id in self.cameras:
            success, message = self.stop_camera(camera_id)
            results.append((camera_id, success, message))
        
        # Release all captures
        for camera_id in list(self.cameras.keys()):
            if self.cameras[camera_id]['cap']:
                self.cameras[camera_id]['cap'].release()
        
        return results
    
    def remove_camera(self, camera_id):
        """Remove a camera from the manager"""
        if camera_id not in self.cameras:
            return False, f"Camera {camera_id} not found"
        
        # Stop the camera if it's running
        if self.cameras[camera_id]['running']:
            self.stop_camera(camera_id)
        
        # Release the capture
        if self.cameras[camera_id]['cap']:
            self.cameras[camera_id]['cap'].release()
        
        # Remove from dictionaries
        del self.cameras[camera_id]
        del self.frame_queues[camera_id]
        del self.frame_buffers[camera_id]
        
        return True, f"Camera {camera_id} removed"
    
    def get_camera_ids(self):
        """Get list of camera IDs"""
        return list(self.cameras.keys())
    
    def _capture_frames(self, camera_id):
        """Continuously capture frames from a camera (run in a thread)"""
        cap = self.cameras[camera_id]['cap']
        frame_queue = self.frame_queues[camera_id]
        frame_buffer = self.frame_buffers[camera_id]
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Minimize buffer size to reduce latency
        
        last_frame_time = time.time()
        consecutive_errors = 0
        
        while self.cameras[camera_id]['running']:
            try:
                # Read a new frame
                ret, frame = cap.read()
                
                if not ret:
                    # Handle read errors
                    consecutive_errors += 1
                    if consecutive_errors > 5:
                        # Try to reopen the camera after multiple failures
                        cap.release()
                        time.sleep(0.5)
                        cap = cv2.VideoCapture(camera_id)
                        self.cameras[camera_id]['cap'] = cap
                        consecutive_errors = 0
                    time.sleep(0.1)
                    continue
                
                consecutive_errors = 0
                current_time = time.time()
                
                # Store the frame
                self.cameras[camera_id]['frame'] = frame
                
                # Try to add to the frame queue for processing
                try:
                    frame_queue.put_nowait((frame, current_time))
                except queue.Full:
                    # If queue is full, skip this frame
                    pass
                
                # Add to the smoothing buffer
                frame_buffer.append(frame)
                
                # Calculate appropriate sleep time to maintain desired FPS
                # Aim for 30 FPS (33.3ms per frame) but adjust based on actual processing time
                elapsed = current_time - last_frame_time
                sleep_time = max(0, (1/30) - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                last_frame_time = time.time()
                
            except Exception as e:
                print(f"Error capturing frame from camera {camera_id}: {str(e)}")
                time.sleep(0.1)
        
        print(f"Stopping capture for camera {camera_id}")


class PeopleTrackingGUI:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Configure main window
        self.window.configure(bg='#f0f0f0')
        self.window.geometry('1200x800')
        
        # Create a style
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TButton', padding=6)
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        
        # Create main frames
        self.main_frame = ttk.Frame(window)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Split into left (camera) and right (controls) panes
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Create camera frame grid
        self.camera_frame = ttk.Frame(self.left_frame)
        self.camera_frame.pack(fill=tk.BOTH, expand=True)
        
        # Controls frame
        self.control_frame = ttk.LabelFrame(self.right_frame, text="Controls")
        self.control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create buttons
        self.btn_start = ttk.Button(self.control_frame, text="Start Tracking", command=self.start_tracking)
        self.btn_start.pack(pady=5, fill=tk.X)
        
        self.btn_stop = ttk.Button(self.control_frame, text="Stop Tracking", command=self.stop_tracking)
        self.btn_stop.pack(pady=5, fill=tk.X)
        self.btn_stop.config(state=tk.DISABLED)
        
        # Extract Data button
        self.btn_extract = ttk.Button(self.control_frame, text="Extract Detection Time Data", command=self.extract_time_data)
        self.btn_extract.pack(pady=5, fill=tk.X)
        
        # Camera Management frame
        camera_mgmt_frame = ttk.LabelFrame(self.right_frame, text="Camera Management")
        camera_mgmt_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Camera selection
        camera_entry_frame = ttk.Frame(camera_mgmt_frame)
        camera_entry_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(camera_entry_frame, text="Camera ID:").pack(side=tk.LEFT, padx=5)
        self.camera_var = tk.StringVar(value="0")
        camera_entry = ttk.Entry(camera_entry_frame, textvariable=self.camera_var, width=10)
        camera_entry.pack(side=tk.LEFT, padx=5)
        
        # Add/Remove camera buttons
        camera_btn_frame = ttk.Frame(camera_mgmt_frame)
        camera_btn_frame.pack(fill=tk.X, pady=5)
        
        self.btn_add_camera = ttk.Button(camera_btn_frame, text="Add Camera", command=self.add_camera)
        self.btn_add_camera.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.btn_remove_camera = ttk.Button(camera_btn_frame, text="Remove Camera", command=self.remove_camera)
        self.btn_remove_camera.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Camera list
        ttk.Label(camera_mgmt_frame, text="Active Cameras:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        
        self.camera_listbox = tk.Listbox(camera_mgmt_frame, height=4)
        self.camera_listbox.pack(fill=tk.X, padx=5, pady=5)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(self.right_frame, text="Settings")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Confidence threshold slider
        ttk.Label(settings_frame, text="Detection Confidence:").pack(anchor=tk.W, padx=5, pady=2)
        self.conf_threshold = tk.DoubleVar(value=0.4) # Slightly lower default might be okay
        conf_slider = ttk.Scale(settings_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, 
                               variable=self.conf_threshold, length=200)
        conf_slider.pack(anchor=tk.W, padx=5, pady=2)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(self.right_frame, text="Statistics")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.people_count_var = tk.StringVar(value="People count: 0")
        ttk.Label(stats_frame, textvariable=self.people_count_var).pack(anchor=tk.W, padx=5, pady=5)
        
        self.fps_var = tk.StringVar(value="FPS: 0")
        ttk.Label(stats_frame, textvariable=self.fps_var).pack(anchor=tk.W, padx=5, pady=5)
        
        # Log frame (at bottom of right frame as requested)
        self.log_frame = ttk.LabelFrame(self.right_frame, text="Activity Log")
        self.log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(self.log_frame, height=10, width=40) # Reduced height slightly
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar for log text
        log_scrollbar = ttk.Scrollbar(self.log_frame, command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        
        # Initialize variables
        self.camera_manager = CameraManager()
        self.is_tracking = False
        self.trackers = {}  # {camera_id: PersonTracker}
        self.camera_canvases = {}  # {camera_id: canvas}
        self.camera_photos = {}  # {camera_id: PhotoImage} - Store PhotoImage refs
        
        # Processing thread and display update loop
        self.processing_thread = None
        self.display_update_ms = 50 # Update display every ~50ms (20 FPS target)
        self._display_after_id = None # To cancel pending display updates
        
        # Protocol for window closing
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.log_message("Application started. Add cameras and click 'Start Tracking'.")
        self.log_message(f"Using device: {device}")
        if DeepFace is None:
            self.log_message("WARNING: DeepFace library failed to load. Face Re-ID disabled.")
        
        # Add default camera
        self.add_camera() # This calls setup_camera_grid
    
    def add_camera(self):
        """Add a new camera based on the camera_var entry"""
        camera_id_str = self.camera_var.get()
        camera_id = camera_id_str # Keep as string if not purely numeric

        try:
            # Try converting to int only if it's all digits
            if camera_id_str.isdigit():
                camera_id = int(camera_id_str)
        except ValueError:
             messagebox.showerror("Error", f"Invalid Camera ID format: {camera_id_str}")
             return
        
        # Check if camera already exists
        if camera_id in self.camera_manager.get_camera_ids():
            messagebox.showinfo("Info", f"Camera '{camera_id}' is already added")
            return
        
        # Add camera to manager
        self.log_message(f"Attempting to add camera '{camera_id}'...")
        success, message = self.camera_manager.add_camera(camera_id)
        if not success:
            messagebox.showerror("Error", message)
            self.log_message(f"Failed to add camera '{camera_id}': {message}")
            return
        
        # Add to UI list
        self.camera_listbox.insert(tk.END, f"Camera {camera_id}")
        
        # Create a tracker for this camera
        self.trackers[camera_id] = PersonTracker()
        
        # Recreate the camera grid when cameras change
        self.setup_camera_grid()
        
        self.log_message(f"Successfully added camera '{camera_id}'")
    
    def remove_camera(self):
        """Remove selected camera from listbox"""
        selection = self.camera_listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a camera to remove")
            return
        
        # Get camera ID from listbox text
        camera_text = self.camera_listbox.get(selection[0])
        try:
            camera_id_str = camera_text.split("Camera ")[1]
            # Convert back to int if needed
            camera_id = int(camera_id_str) if camera_id_str.isdigit() else camera_id_str
        except (IndexError, ValueError):
            messagebox.showerror("Error", f"Could not parse camera ID from '{camera_text}'")
            return
        
        self.log_message(f"Removing camera '{camera_id}'...")
        # Remove from manager
        success, message = self.camera_manager.remove_camera(camera_id)
        if not success:
            messagebox.showerror("Error", message)
            self.log_message(f"Error removing camera '{camera_id}': {message}")
            # Continue cleanup even if manager removal had issues
        
        # Remove from UI list
        self.camera_listbox.delete(selection[0])
        
        # Remove tracker
        if camera_id in self.trackers:
            del self.trackers[camera_id]
        
        # Remove canvas if it exists
        if camera_id in self.camera_canvases:
             self.camera_canvases[camera_id].destroy()
             del self.camera_canvases[camera_id]
        if camera_id in self.camera_photos:
             del self.camera_photos[camera_id]
        
        # Recreate the camera grid
        self.setup_camera_grid()
        
        self.log_message(f"Removed camera '{camera_id}'")
    
    def setup_camera_grid(self):
        """Set up the camera display grid based on number of cameras"""
        # Clear existing camera frames/canvases first
        for widget in self.camera_frame.winfo_children():
            widget.destroy()
        
        # Clear canvas references (photos are cleared in remove_camera or on_close)
        self.camera_canvases = {}
        
        camera_ids = self.camera_manager.get_camera_ids()
        if not camera_ids:
            msg_label = ttk.Label(self.camera_frame, text="No cameras added. Add a camera to begin.")
            msg_label.pack(expand=True)
            self.camera_frame.rowconfigure(0, weight=1) # Ensure message is centered
            self.camera_frame.columnconfigure(0, weight=1)
            return
        
        # Calculate grid dimensions (e.g., max 2 columns)
        n_cameras = len(camera_ids)
        cols = min(2, n_cameras)
        rows = (n_cameras + cols - 1) // cols
        
        # Create canvases in the grid
        for i, camera_id in enumerate(camera_ids):
            row = i // cols
            col = i % cols
            
            # Frame to hold canvas and label
            camera_container = ttk.LabelFrame(self.camera_frame, text=f"Camera {camera_id}")
            camera_container.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            self.camera_frame.rowconfigure(row, weight=1)
            self.camera_frame.columnconfigure(col, weight=1)
            
            # Canvas for video display
            canvas = tk.Canvas(camera_container, bg="black", width=320, height=240) # Initial size hint
            canvas.pack(fill=tk.BOTH, expand=True)
            self.camera_canvases[camera_id] = canvas
    
    def start_tracking(self):
        if self.is_tracking:
            return
        
        camera_ids = self.camera_manager.get_camera_ids()
        if not camera_ids:
            messagebox.showinfo("Info", "No cameras added. Please add at least one camera.")
            return
        
        if model is None:
             messagebox.showerror("Error", "YOLO model not loaded. Cannot start tracking.")
             return
        if DeepFace is None:
             # Warn but allow continuing without face re-id
             messagebox.showwarning("Warning", "DeepFace library not loaded. Face Re-ID will be disabled.")


        # Start all cameras first
        self.log_message("Starting cameras...")
        start_results = self.camera_manager.start_all_cameras()
        failures = [(cam_id, msg) for cam_id, success, msg in start_results if not success]
        
        if failures:
            error_msg = "Failed to start some cameras:\n" + "\n".join([f"- {cam_id}: {msg}" for cam_id, msg in failures])
            messagebox.showerror("Error", error_msg)
            self.log_message(error_msg)
            # Stop cameras that did start successfully
            successful_starts = [cam_id for cam_id, success, _ in start_results if success]
            for cam_id in successful_starts:
                 self.camera_manager.stop_camera(cam_id)
            return
        
        self.is_tracking = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_add_camera.config(state=tk.DISABLED) # Disable add/remove during tracking
        self.btn_remove_camera.config(state=tk.DISABLED)
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_frames_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start display update loop
        self.update_display_loop() # Start the periodic GUI update

        self.log_message(f"Tracking started with {len(camera_ids)} cameras.")
    
    def stop_tracking(self):
        if not self.is_tracking:
            return

        self.is_tracking = False # Signal processing loop to stop

        # Cancel any pending display updates
        if self._display_after_id:
             self.window.after_cancel(self._display_after_id)
             self._display_after_id = None

        # Wait for processing thread to finish (add a timeout)
        if self.processing_thread and self.processing_thread.is_alive():
             self.log_message("Waiting for processing thread to finish...")
             self.processing_thread.join(timeout=2.0) # Wait max 2 seconds
             if self.processing_thread.is_alive():
                 self.log_message("Warning: Processing thread did not exit cleanly.")

        # Stop all cameras (releases resources)
        self.log_message("Stopping cameras...")
        self.camera_manager.stop_all_cameras()
        
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_add_camera.config(state=tk.NORMAL)
        self.btn_remove_camera.config(state=tk.NORMAL)

        # Reset FPS counter
        self.fps_var.set("FPS: 0")
        self.people_count_var.set("People count: 0")

        self.log_message("Tracking stopped.")


    def process_frames_loop(self):
        """Dedicated thread for processing frames from all cameras."""
        frame_count_total = 0
        start_time_total = time.time()
        last_fps_update_time = start_time_total
        
        while self.is_tracking:
            current_loop_start_time = time.time()
            processed_this_loop = False

            camera_ids = self.camera_manager.get_camera_ids()
            active_people_counts = {} # Store counts per camera {cam_id: count}

            for camera_id in camera_ids:
                # Get latest frame from the buffer
                success, frame, error = self.camera_manager.get_frame(camera_id)

                if not success or frame is None:
                    # Optional: log if frame retrieval fails often
                    # print(f"Skipping frame for camera {camera_id}, not available.")
                    continue # Skip to next camera if no frame
                
                processed_this_loop = True
                try:
                    # --- Core Processing ---
                    # 1. Detect people (YOLO)
                    detections = self.detect_people(frame)
                    
                    # 2. Update tracker (IoU + Face Re-ID)
                    tracker = self.trackers[camera_id]
                    tracked_objects = tracker.update(frame, detections) # Pass frame for feature extraction
                    
                    # 3. Draw results (on a copy)
                    result_frame = self.draw_results(frame, tracked_objects)
                    
                    # 4. Store result frame for display update
                    # We update the display in a separate loop to avoid blocking this thread
                    # For simplicity here, we'll call display_frame directly, but it can lag the processing.
                    # A better way is queueing frames for display update loop.
                    # self.display_frame(camera_id, result_frame) # Direct update (can lag)

                    # Store the frame to be displayed later by the update_display_loop
                    # Need a mechanism to store these... maybe a shared dict?
                    # For now, let's update directly for simplicity, but acknowledge the lag risk.


                    # --- Statistics ---
                    active_count = sum(1 for obj in tracked_objects.values() if obj.get('active', False))
                    active_people_counts[camera_id] = active_count
                    frame_count_total += 1
                    
                except Exception as e:
                    self.log_message(f"Error processing frame from camera {camera_id}: {str(e)}")
                    # Optionally display an error indicator on the canvas?

            # --- Update Global Stats periodically ---
            current_time = time.time()
            if current_time - last_fps_update_time >= 1.0:
                elapsed_total = current_time - start_time_total
                if elapsed_total > 0:
                    fps = frame_count_total / elapsed_total
                self.fps_var.set(f"FPS: {fps:.2f}")

                # Update total people count (sum across cameras)
                total_people = sum(active_people_counts.values())
                self.people_count_var.set(f"People count: {total_people}")

                # Reset for next interval
                start_time_total = current_time
                frame_count_total = 0
                last_fps_update_time = current_time


            # --- Control loop speed ---
            # Avoid busy-waiting if no frames were processed
            if not processed_this_loop:
                 time.sleep(0.05) # Sleep longer if no cameras active/returning frames
            else:
                 # Add a small sleep to yield CPU, prevent 100% usage
                 loop_duration = time.time() - current_loop_start_time
                 sleep_time = max(0.005, 0.01 - loop_duration) # Aim for ~100Hz loop max, min 5ms sleep
                 time.sleep(sleep_time)


    def update_display_loop(self):
         """Periodically updates the GUI canvases with the latest processed frames."""
         if not self.is_tracking:
             return # Stop loop if tracking stopped

         update_start_time = time.time()

         camera_ids = self.camera_manager.get_camera_ids()
         for camera_id in camera_ids:
             if camera_id not in self.camera_canvases: continue # Skip if canvas removed

             # Get the *latest* available frame (might be slightly behind processing)
             success, frame_to_display, _ = self.camera_manager.get_frame(camera_id)

             if success and frame_to_display is not None:
                 # Re-run detection and drawing for display? No, use processed frame.
                 # Problem: process_frames_loop doesn't store processed frames efficiently for this loop.
                 # --- TEMPORARY WORKAROUND: Re-process frame for display ---
                 # This is INEFFICIENT. A proper solution needs a shared structure
                 # (like a dictionary mapping camera_id to latest processed_frame).
                 try:
                     tracker = self.trackers[camera_id]
                     # Get current state without updating tracker again
                     tracked_objects_state = tracker.tracked_objects # Get current dict
                     display_frame = self.draw_results(frame_to_display, tracked_objects_state)
                     self.display_frame(camera_id, display_frame)
                 except Exception as e:
                     self.log_message(f"Error preparing display frame for {camera_id}: {e}")
                     # Optionally display the raw frame on error
                     # self.display_frame(camera_id, frame_to_display)
             # else: Keep last displayed frame or show placeholder?


         # Schedule the next update
         # Adjust delay based on how long this update took?
         update_duration = time.time() - update_start_time
         next_update_delay = max(10, self.display_update_ms - int(update_duration * 1000)) # Min 10ms delay

         self._display_after_id = self.window.after(next_update_delay, self.update_display_loop)

    
    def detect_people(self, frame):
        """Detect people in a frame using YOLOv8"""
        detections = []
        if model is None: return detections
        
        try:
            # Ensure frame is in expected format (e.g., BGR numpy array)
        # Run inference
            results = model(frame, conf=self.conf_threshold.get(), classes=[0], verbose=False, device=device) # class 0 is person, disable verbose logging

            if results and len(results) > 0:
                result = results[0] # Get Boxes object for the first image
                boxes = result.boxes.cpu().numpy() # Get boxes in numpy format (xyxy, conf, cls)

            for box in boxes:
                    if int(box.cls[0]) == 0: # Check class ID is 0 (person)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                    detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf
                    })
        except Exception as e:
             self.log_message(f"Error during YOLO detection: {e}")
             # Optionally re-raise or return empty list
        
        return detections
    
    
    def draw_results(self, frame, tracked_objects):
        """Draw bounding boxes and IDs on the frame"""
        result = frame.copy()
        current_time = time.time()
        
        for obj_id, obj_data in tracked_objects.items():
            if not obj_data.get('active', True):
                continue # Don't draw inactive tracks

            x1, y1, x2, y2 = map(int, obj_data['bbox'])
            
            # Calculate active time from intervals instead of first_seen
            active_time = 0
            if obj_id in self.trackers[0].time_data:  # Assuming camera 0 for now
                time_info = self.trackers[0].time_data[obj_id]
                active_intervals = time_info.get('active_intervals', [])
                
                for interval in active_intervals:
                    start = interval[0]
                    end = interval[1] if interval[1] is not None else current_time
                    active_time += end - start

            time_str = self._format_duration(active_time)

            # Draw bounding box (Green for active)
            color = (0, 255, 0)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Calculate center position for text
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Draw ID and time at center
            label = f"ID: {obj_id}\n{time_str}"
            if obj_data.get('feature') is not None:
                label += " [F]"  # Indicate feature present

            # Split text into lines
            lines = label.split('\n')
            
            # Get text sizes
            font_scale = 0.7
            font = cv2.FONT_HERSHEY_SIMPLEX
            line_heights = [cv2.getTextSize(line, font, font_scale, 2)[0][1] + 5 for line in lines]
            total_height = sum(line_heights)
            
            # Draw each line centered
            y = center_y - total_height//2
            for i, line in enumerate(lines):
                text_size = cv2.getTextSize(line, font, font_scale, 2)[0]
                text_x = center_x - text_size[0]//2
                
                # Draw background rectangle
                bg_pad = 5
                cv2.rectangle(result, 
                             (text_x - bg_pad, int(y - bg_pad)), 
                             (text_x + text_size[0] + bg_pad, int(y + text_size[1] + bg_pad)), 
                             color, cv2.FILLED)
                
                # Draw text
                cv2.putText(result, line, (text_x, int(y + text_size[1])), 
                           font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
                
                y += line_heights[i]
        
        return result
    
    def display_frame(self, camera_id, frame):
        """Display a frame on the appropriate canvas"""
        if camera_id not in self.camera_canvases:
            # print(f"Canvas for camera {camera_id} not found for display.")
            return # Canvas might have been removed
            
        canvas = self.camera_canvases[camera_id]
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # If canvas dimensions are invalid (e.g., during init), use default or skip
        if canvas_width <= 1 or canvas_height <= 1:
             # print(f"Canvas {camera_id} not ready for display (width={canvas_width}, height={canvas_height})")
             return
        
        try:
        # Calculate aspect ratio preserving resize
            frame_height, frame_width = frame.shape[:2]
            if frame_width == 0 or frame_height == 0: return # Skip empty frame

            scale = min(canvas_width / frame_width, canvas_height / frame_height)
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
        
            # Resize frame smoothly
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
            # Convert BGR to RGB for PIL -> Tkinter
            img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            photo = ImageTk.PhotoImage(image=img_pil)
        
            # Store reference to prevent garbage collection (!IMPORTANT!)
            self.camera_photos[camera_id] = photo
        
            # Display on canvas (center image)
            # canvas.delete("all") # Clear previous image
            # canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo, anchor=tk.CENTER)

            # More efficient update: find existing image item and update its data
            img_item = canvas.find_withtag("frame_img")
            if img_item:
                canvas.itemconfig(img_item, image=photo)
            else:
                canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo, anchor=tk.CENTER, tags="frame_img")


        except Exception as e:
            self.log_message(f"Error displaying frame for camera {camera_id}: {e}")

    
    def extract_time_data(self):
        """Extract time data for all tracked people"""
        try:
            # Collect time data from all trackers
            all_data = []
            for camera_id, tracker in self.trackers.items():
                try:
                    camera_data = tracker.get_time_data()
                    # Add camera ID to each entry
                    for entry in camera_data:
                        entry['camera_id'] = camera_id
                    all_data.extend(camera_data)
                except Exception as e:
                     self.log_message(f"Error getting time data from tracker for camera {camera_id}: {e}")
            
            if not all_data:
                messagebox.showinfo("Info", "No tracking data available to export.")
                return
            
            # Ask user for save location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save Tracking Time Data As"
            )
            
            if not file_path:
                return  # User cancelled
            
            # Write data to CSV
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                # Ensure all potential keys are included
                fieldnames = ['camera_id', 'id', 'first_seen', 'last_seen', 'duration', 'duration_seconds', 'status']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore') # Ignore extra keys if any
                
                writer.writeheader()
                # Sort data before writing? Maybe by camera then ID?
                all_data.sort(key=lambda x: (x.get('camera_id', ''), x.get('id', 0)))
                for entry in all_data:
                    writer.writerow(entry)
            
            self.log_message(f"Time data exported to {file_path}")
            messagebox.showinfo("Success", f"Time data successfully exported to\n{file_path}")
            
        except Exception as e:
            self.log_message(f"Error exporting time data: {str(e)}")
            messagebox.showerror("Error", f"Failed to export time data: {str(e)}")
    
    def log_message(self, message):
        """Add a message to the log with timestamp (thread-safe using schedule)"""
        def append_log():
             if self.log_text.winfo_exists(): # Check if widget still exists
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3] # Add milliseconds
                log_entry = f"[{timestamp}] {message}\n"
                self.log_text.insert(tk.END, log_entry)
                self.log_text.see(tk.END) # Auto-scroll
                # Print to console as well
                # print(log_entry.strip())

        # Schedule the GUI update from the main thread
        if hasattr(self.window, 'after'):
             try:
                 self.window.after(0, append_log)
             except tk.TclError: # Handle case where window is destroyed
                 print(f"Log (window closed): {message}")
        else: # Fallback if window object is not fully initialized or in weird state
            print(f"Log (fallback): {message}")

    
    def on_close(self):
        """Handle window closing"""
        self.log_message("Close button clicked. Shutting down...")
        # Stop tracking first
        if self.is_tracking:
            self.stop_tracking() # This now handles stopping cameras and joining threads
        
        # Ensure all cameras are released (belt-and-suspenders)
        # self.camera_manager.stop_all_cameras() # Already called in stop_tracking
        
        self.log_message("Destroying GUI window.")
        # Close the window
        self.window.destroy()

    def _format_duration(self, seconds):
        """Formats seconds into hours:minutes:seconds format"""
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = PeopleTrackingGUI(root, "People Tracking System with Face Re-ID")
        root.mainloop()
    except Exception as e:
        print(f"Fatal error in main execution: {e}")
        import traceback
        traceback.print_e
                                        
                                                                                                                                                                                                                                                                                            
