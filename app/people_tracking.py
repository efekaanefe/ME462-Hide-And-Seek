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
import os
import json

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
        self.reid_time_window = 1000.0  # Very long to effectively keep all tracks for re-id
        # self.iou_threshold = 0.3
        self.feature_threshold = 0.9  # Base threshold for new identifications
        self.reidentification_threshold = 0.9  # More lenient threshold for recent tracks
        self.face_model_name = "Facenet"
        self.face_detector_backend = "mtcnn"
        
        # Known people database
        self.known_people_dir = "known_people"
        self.known_people = {}  # {name: {'features': [feature_vectors], 'images': [image_paths]}}
        # New: Track mapping for known people
        self.known_people_tracks = {}  # {name: [track_ids]}
        self.load_known_people()
        
        # Add total_active_time to separate from elapsed time
        self.time_data = {}  # {id: {'first_seen': timestamp, 'last_seen': timestamp, 
                            #       'total_active_time': seconds, 'active_intervals': [(start, end), ...]}
        
        # New structure to temporarily track unidentified people (no face detected yet)
        self.unidentified_tracks = {}  # {temp_id: {'bbox': bbox, 'first_seen': timestamp, 'last_seen': timestamp}}
        self.temp_id_counter = 1  # Counter for temporary IDs
        
        # New: Add feature history for more stable identification
        self.feature_history = {}  # {id: [list of recent features]}
        self.max_feature_history = 5  # Keep last 5 features for each ID
        
        if DeepFace is None:
             print("WARNING: DeepFace library not available. Face re-identification will be disabled.")

    def load_known_people(self):
        """Load known people dataset from directory structure."""
        if not os.path.exists(self.known_people_dir):
            os.makedirs(self.known_people_dir)
            print(f"Created known people directory: {self.known_people_dir}")
            return

        print("Loading known people dataset...")
        for person_name in os.listdir(self.known_people_dir):
            person_dir = os.path.join(self.known_people_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            features = []
            image_paths = []
            
            # Process each image in person's directory
            for img_file in os.listdir(person_dir):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                img_path = os.path.join(person_dir, img_file)
                try:
                    # Extract face feature from the image
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Could not read image {img_path}")
                        continue
                        
                    feature = self._extract_face_feature(img, [0, 0, img.shape[1], img.shape[0]])
                    if feature is not None:
                        features.append(feature)
                        image_paths.append(img_path)
                    else:
                        print(f"Warning: Could not extract face feature from {img_path}")
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

            if features:
                self.known_people[person_name] = {
                    'features': features,
                    'images': image_paths
                }
                print(f"Loaded {len(features)} features for {person_name}")
            else:
                print(f"Warning: No valid features extracted for {person_name}")

    def add_person_images(self, name, image_paths):
        """Add new images for a person to the dataset."""
        person_dir = os.path.join(self.known_people_dir, name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        features = []
        saved_paths = []
        
        for img_path in image_paths:
            try:
                # Read and process the image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                
                # Extract face feature
                feature = self._extract_face_feature(img, [0, 0, img.shape[1], img.shape[0]])
                if feature is not None:
                    # Copy image to person's directory
                    new_path = os.path.join(person_dir, os.path.basename(img_path))
                    cv2.imwrite(new_path, img)
                    
                    features.append(feature)
                    saved_paths.append(new_path)
                else:
                    print(f"Warning: Could not extract face feature from {img_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Update known people database
        if features:
            if name not in self.known_people:
                self.known_people[name] = {'features': [], 'images': []}
            
            self.known_people[name]['features'].extend(features)
            self.known_people[name]['images'].extend(saved_paths)
            print(f"Added {len(features)} new features for {name}")
        
        return len(features)

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

    def _update_feature_history(self, track_id, new_feature):
        """Update feature history for a track and compute average feature"""
        if track_id not in self.feature_history:
            self.feature_history[track_id] = []
        
        # Add new feature to history
        self.feature_history[track_id].append(new_feature)
        
        # Keep only recent features
        if len(self.feature_history[track_id]) > self.max_feature_history:
            self.feature_history[track_id].pop(0)
        
        # Compute average feature
        avg_feature = np.mean(self.feature_history[track_id], axis=0)
        return avg_feature

    def _find_best_face_match(self, face_feature, current_time):
        """Find the best matching face in the database with improved matching logic"""
        best_match_id = -1
        min_distance = float('inf')
        best_name = "UNK"  # Default to unknown
        
        # First, try to match with known people
        for name, data in self.known_people.items():
            for ref_feature in data['features']:
                distance = self._calculate_feature_distance(face_feature, ref_feature)
                if distance < self.feature_threshold and distance < min_distance:
                    min_distance = distance
                    best_name = name
                    # Look for existing track with this name to maintain duration
                    for track_id, track_data in self.tracked_objects.items():
                        if track_data.get('name') == name and track_data.get('active', False):
                            best_match_id = track_id
                            break
        
        # If no match in known people, try active tracks for re-identification
        if best_name == "UNK":
            for db_id, db_entry in self.face_database.items():
                if 'feature' not in db_entry or db_entry['feature'] is None:
                    continue
                
                # Calculate time since last seen
                time_since_last_seen = current_time - db_entry['last_seen']
                
                # Use different thresholds based on recency
                threshold = self.reidentification_threshold if time_since_last_seen < 5.0 else self.feature_threshold
                
                # Compare with average feature if available
                if db_id in self.feature_history:
                    avg_feature = np.mean(self.feature_history[db_id], axis=0)
                    distance = self._calculate_feature_distance(face_feature, avg_feature)
                else:
                    distance = self._calculate_feature_distance(face_feature, db_entry['feature'])
                
                if distance < threshold and distance < min_distance:
                    min_distance = distance
                    best_match_id = db_id
        
        return best_match_id, best_name, min_distance

    def update(self, frame, detections, current_time=None):
        """Update tracks using Face Features only with improved stability."""
        if current_time is None:
            current_time = time.time()
        
        active_objects = {}
        matched_track_ids = set()
        matched_temp_ids = set()
        newly_detected_indices = set(range(len(detections)))
        
        # --- STEP 1: First try to match active permanent tracks by face ---
        active_permanent_tracks = {id: data for id, data in self.tracked_objects.items() 
                                 if data.get('active', True) and not isinstance(id, str)}
        
        if active_permanent_tracks and newly_detected_indices:
            for track_id, track_data in active_permanent_tracks.items():
                if not track_data.get('active', True):
                    continue
                
                # Try to find the best matching detection for this track
                best_det_idx = -1
                best_distance = float('inf')
                
                for det_idx in newly_detected_indices:
                    bbox = detections[det_idx]['bbox']
                    face_feature = self._extract_face_feature(frame, bbox)
                    
                    if face_feature is not None and track_id in self.feature_history:
                        # Compare with average feature
                        avg_feature = np.mean(self.feature_history[track_id], axis=0)
                        distance = self._calculate_feature_distance(face_feature, avg_feature)
                        
                        if distance < self.reidentification_threshold and distance < best_distance:
                            best_distance = distance
                            best_det_idx = det_idx
                
                if best_det_idx != -1:
                    # Update track with new detection
                    bbox = detections[best_det_idx]['bbox']
                    face_feature = self._extract_face_feature(frame, bbox)
                    
                    self.tracked_objects[track_id]['bbox'] = bbox
                    self.tracked_objects[track_id]['last_seen'] = current_time
                    
                    # Update feature history
                    if face_feature is not None:
                        avg_feature = self._update_feature_history(track_id, face_feature)
                        self.face_database[track_id]['feature'] = avg_feature
                        self.face_database[track_id]['last_seen'] = current_time
                    
                    # Ensure time tracking continues
                    if track_id in self.time_data:
                        self.time_data[track_id]['last_seen'] = current_time
                        intervals = self.time_data[track_id]['active_intervals']
                        if not intervals or intervals[-1][1] is not None:
                            intervals.append([current_time, None])
                    
                    matched_track_ids.add(track_id)
                    newly_detected_indices.remove(best_det_idx)
        
        # --- STEP 2: Process remaining detections ---
        for det_idx in list(newly_detected_indices):
            bbox = detections[det_idx]['bbox']
            face_feature = self._extract_face_feature(frame, bbox)
            
            if face_feature is not None:
                best_match_id, name, min_distance = self._find_best_face_match(face_feature, current_time)
                
                if best_match_id != -1:
                    # Update existing track
                    self.tracked_objects[best_match_id]['bbox'] = bbox
                    self.tracked_objects[best_match_id]['last_seen'] = current_time
                    self.tracked_objects[best_match_id]['active'] = True
                    self.tracked_objects[best_match_id]['name'] = name
                    
                    # Update known people tracks mapping
                    if name != "UNK" and name not in self.known_people_tracks:
                        self.known_people_tracks[name] = []
                    if name != "UNK" and best_match_id not in self.known_people_tracks[name]:
                        self.known_people_tracks[name].append(best_match_id)
                    
                    # Update feature history and database
                    avg_feature = self._update_feature_history(best_match_id, face_feature)
                    self.face_database[best_match_id]['feature'] = avg_feature
                    self.face_database[best_match_id]['last_seen'] = current_time
                    
                    # Ensure time tracking continues
                    if best_match_id in self.time_data:
                        self.time_data[best_match_id]['last_seen'] = current_time
                        intervals = self.time_data[best_match_id]['active_intervals']
                        if not intervals or intervals[-1][1] is not None:
                            intervals.append([current_time, None])
                    
                    matched_track_ids.add(best_match_id)
                else:
                    # Create new track
                    new_id = self.next_id
                    self.next_id += 1
                    
                    self.tracked_objects[new_id] = {
                        'bbox': bbox,
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'active': True,
                        'name': name
                    }
                    
                    # Update known people tracks mapping for new track
                    if name != "UNK":
                        if name not in self.known_people_tracks:
                            self.known_people_tracks[name] = []
                        self.known_people_tracks[name].append(new_id)
                        print(f"New track {new_id} created for known person {name} (Total tracks: {len(self.known_people_tracks[name])})")
                    else:
                        print(f"New track created for unknown person (ID: {new_id})")
                    
                    # Initialize feature history and database
                    self._update_feature_history(new_id, face_feature)
                    self.face_database[new_id] = {
                        'feature': face_feature,
                        'first_seen': current_time,
                        'last_seen': current_time
                    }
                    
                    # Initialize time tracking
                    self.time_data[new_id] = {
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'total_active_time': 0,
                        'active_intervals': [[current_time, None]]
                    }
                    
                    matched_track_ids.add(new_id)
        
        # --- Update status of existing tracks ---
        for obj_id in list(self.tracked_objects.keys()):
            if obj_id not in matched_track_ids:
                if self.tracked_objects[obj_id].get('active', True):
                    if current_time - self.tracked_objects[obj_id]['last_seen'] > self.disappear_threshold:
                        self.tracked_objects[obj_id]['active'] = False
                        
                        # Close the active interval
                        if obj_id in self.time_data:
                            intervals = self.time_data[obj_id]['active_intervals']
                            if intervals and intervals[-1][1] is None:
                                intervals[-1][1] = current_time
                                interval_duration = current_time - intervals[-1][0]
                                self.time_data[obj_id]['total_active_time'] += interval_duration
                                self.time_data[obj_id]['last_seen'] = current_time
        
        # Return combined tracks for visualization
        combined_tracks = self.tracked_objects.copy()
        for temp_id, temp_data in self.unidentified_tracks.items():
            temp_track = temp_data.copy()
            temp_track['is_temporary'] = True
            temp_track['active'] = True
            combined_tracks[temp_id] = temp_track
        
        return combined_tracks
    
    def get_time_data(self):
        """Get time data for all tracked objects with consolidated known people information"""
        current_time = time.time()
        result = []
        
        # First, process known people's consolidated data
        for name, track_ids in self.known_people_tracks.items():
            total_active_time = 0
            first_seen = float('inf')
            last_seen = 0
            active_status = False
            track_intervals = []
            
            for track_id in track_ids:
                if track_id in self.time_data:
                    time_info = self.time_data[track_id]
                    # Update overall first/last seen
                    first_seen = min(first_seen, time_info['first_seen'])
                    last_seen = max(last_seen, time_info['last_seen'])
                    
                    # Accumulate intervals
                    for interval in time_info['active_intervals']:
                        start = interval[0]
                        end = interval[1] if interval[1] is not None else (
                            current_time if track_id in self.tracked_objects and 
                            self.tracked_objects[track_id].get('active', False) else start
                        )
                        track_intervals.append((start, end))
                        total_active_time += end - start
                    
                    # Check if any track is currently active
                    if track_id in self.tracked_objects and self.tracked_objects[track_id].get('active', False):
                        active_status = True
            
            if first_seen != float('inf'):  # Only add if we have valid data
                result.append({
                    'name': name,
                    'track_ids': track_ids,
                    'first_seen': datetime.fromtimestamp(first_seen).strftime("%Y-%m-%d %H:%M:%S"),
                    'last_seen': datetime.fromtimestamp(last_seen).strftime("%Y-%m-%d %H:%M:%S"),
                    'duration': self._format_duration(total_active_time),
                    'duration_seconds': total_active_time,
                    'status': 'Active' if active_status else 'Inactive',
                    'is_known': True
                })
        
        # Then process unknown tracks (not associated with known people)
        for obj_id, time_info in self.time_data.items():
            # Skip if this track belongs to a known person
            skip = False
            for track_ids in self.known_people_tracks.values():
                if obj_id in track_ids:
                    skip = True
                    break
            if skip:
                continue
            
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
                'status': 'Active' if active else 'Inactive',
                'is_known': False
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
            
            # Check if this is a temporary track (awaiting face detection)
            is_temporary = obj_data.get('is_temporary', False)
            name = obj_data.get('name', 'UNK')
            
            if is_temporary:
                # Yellow color for temporary tracks (waiting for face)
                color = (0, 255, 255)  # Yellow in BGR
                
                # For temporary tracks, calculate time since first seen
                if 'first_seen' in obj_data:
                    time_visible = current_time - obj_data['first_seen']
                    time_str = self._format_duration(time_visible)
                else:
                    time_str = "00:00:00"
                
                # Display as temporary ID
                label = f"Waiting for face\n{time_str}"
            else:
                # Regular track with permanent ID
                # Calculate active time from intervals
                active_time = 0
                camera_id = list(self.trackers.keys())[0] if self.trackers else 0  # Default to first camera
                
                if obj_id in self.trackers[camera_id].time_data:
                    time_info = self.trackers[camera_id].time_data[obj_id]
                    active_intervals = time_info.get('active_intervals', [])
                    
                    for interval in active_intervals:
                        start = interval[0]
                        end = interval[1] if interval[1] is not None else current_time
                        active_time += end - start

                time_str = self._format_duration(active_time)
                
                # Color based on identification status
                if name == "UNK":
                    # Orange for unknown but detected faces
                    color = (0, 165, 255)  # Orange in BGR
                else:
                    # Green for known people
                    color = (0, 255, 0)  # Green in BGR
                
                # Display name/ID and time
                if name == "UNK":
                    label = f"Unknown #{obj_id}\n{time_str}"
                else:
                    label = f"{name}\n{time_str}"

            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Calculate center position for text
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
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
        """Extract time data for all tracked people in a hierarchical structure"""
        try:
            # Collect raw time data from all trackers
            raw_data = {}  # {camera_id: tracker_data}
            for camera_id, tracker in self.trackers.items():
                try:
                    camera_data = tracker.get_time_data()
                    raw_data[camera_id] = camera_data
                except Exception as e:
                    self.log_message(f"Error getting time data from tracker for camera {camera_id}: {e}")
            
            if not raw_data:
                messagebox.showinfo("Info", "No tracking data available to export.")
                return
            
            # Create hierarchical structure
            structured_data = {
                "known_people": {},
                "unknown_people": []
            }
            
            # Process all cameras' data
            for camera_id, camera_data in raw_data.items():
                # Process known people first
                known_entries = [entry for entry in camera_data if entry.get('is_known', False)]
                for entry in known_entries:
                    name = entry['name']
                    if name not in structured_data["known_people"]:
                        structured_data["known_people"][name] = {
                            "total_duration": entry['duration'],
                            "total_duration_seconds": entry['duration_seconds'],
                            "first_seen": entry['first_seen'],
                            "last_seen": entry['last_seen'],
                            "tracks": []
                        }
                    
                    # Get track details for this person
                    tracker = self.trackers[camera_id]
                    for track_id in entry['track_ids']:
                        if track_id in tracker.time_data:
                            track_data = tracker.time_data[track_id]
                            intervals = track_data['active_intervals']
                            
                            # Process each interval as a separate track entry
                            for interval in intervals:
                                start_time = datetime.fromtimestamp(interval[0]).strftime("%Y-%m-%d %H:%M:%S")
                                # Handle ongoing tracks
                                if interval[1] is None:
                                    end_time = "ongoing"
                                    duration_seconds = time.time() - interval[0]
                                else:
                                    end_time = datetime.fromtimestamp(interval[1]).strftime("%Y-%m-%d %H:%M:%S")
                                    duration_seconds = interval[1] - interval[0]
                                
                                track_entry = {
                                    "camera": camera_id,
                                    "track_id": track_id,
                                    "track_start": start_time,
                                    "track_end": end_time,
                                    "duration": self._format_duration(duration_seconds)
                                }
                                structured_data["known_people"][name]["tracks"].append(track_entry)
                
                # Process unknown people
                unknown_entries = [entry for entry in camera_data if not entry.get('is_known', False)]
                for entry in unknown_entries:
                    track_id = entry['id']
                    if track_id in tracker.time_data:
                        track_data = tracker.time_data[track_id]
                        intervals = track_data['active_intervals']
                        
                        for interval in intervals:
                            start_time = datetime.fromtimestamp(interval[0]).strftime("%Y-%m-%d %H:%M:%S")
                            if interval[1] is None:
                                end_time = "ongoing"
                                duration_seconds = time.time() - interval[0]
                            else:
                                end_time = datetime.fromtimestamp(interval[1]).strftime("%Y-%m-%d %H:%M:%S")
                                duration_seconds = interval[1] - interval[0]
                            
                            unknown_entry = {
                                "id": track_id,
                                "camera": camera_id,
                                "track_start": start_time,
                                "track_end": end_time,
                                "duration": self._format_duration(duration_seconds),
                                "total_duration": entry['duration']
                            }
                            structured_data["unknown_people"].append(unknown_entry)
            
            # Ask user for save location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save Tracking Time Data As"
            )
            
            if not file_path:
                return  # User cancelled
            
            # Write to JSON file
            with open(file_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(structured_data, jsonfile, indent=4)
            
            self.log_message(f"Time data exported to {file_path}")
            
            # Display summary in log
            self.log_message("\nTracking Summary:")
            for name, person_data in structured_data["known_people"].items():
                self.log_message(f"\nKnown Person: {name}")
                self.log_message(f"  Total Duration: {person_data['total_duration']}")
                self.log_message(f"  First Seen: {person_data['first_seen']}")
                self.log_message(f"  Last Seen: {person_data['last_seen']}")
                self.log_message("  Tracks:")
                for track in person_data["tracks"]:
                    self.log_message(f"    Camera {track['camera']}: {track['track_start']} -> {track['track_end']} ({track['duration']})")
            
            if structured_data["unknown_people"]:
                self.log_message("\nUnknown Tracks:")
                for track in structured_data["unknown_people"]:
                    self.log_message(f"  ID {track['id']} (Camera {track['camera']}): {track['track_start']} -> {track['track_end']} ({track['duration']})")
            
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
                                        
                                                                                                                                                                                                                                                                                            
