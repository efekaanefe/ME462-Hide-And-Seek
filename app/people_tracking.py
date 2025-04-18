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
from ultralytics import YOLO
import dlib
import face_recognition  # This uses dlib internally with a more convenient API
from scipy.optimize import linear_sum_assignment  # Added for Hungarian Algorithm

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLOv8 model - using Ultralytics implementation
try:
    model = YOLO('models/yolov8n.pt')  # Load the smallest YOLOv8 model to start with
except ImportError:
    messagebox.showerror("Error", "Please install ultralytics: pip install ultralytics")
    model = None


class PersonTracker:
    def __init__(self):
        self.next_id = 1
        self.tracked_objects = {}  # Active/inactive tracks
        self.face_database = {}  # Permanent storage: {id: {'feature': feature_vector, 'last_seen': timestamp}}
        self.disappear_threshold = 2.0
        self.reid_time_window = 100.0  # Very long to effectively keep all tracks for re-id
        self.iou_threshold = 0.4  # Minimum IoU for matching consideration
        
        # Set distance metric for feature comparison
        self.use_cosine_distance = True  # Default is Euclidean, set to True for cosine
        
        # Single similarity threshold (used for both feature matching and re-identification)
        self.similarity_threshold = 0.55  # Default threshold for Euclidean distance
        self.cosine_similarity_threshold = 0.06  # Default threshold for cosine distance
        
        # Periodic re-identification settings
        self.periodic_reid_enabled = True  # Enable periodic re-identification
        self.periodic_reid_interval = 1.0  # Seconds between re-identification checks
        self.last_reid_checks = {}  # Track last re-identification time for each track
        self.reid_failure_threshold = 1  # Number of failed re-IDs before considering identity switch
        self.reid_failure_counts = {}  # Count re-ID failures for each track
        
        # Track identity changes history
        self.identity_changes = {}  # {track_id: [{'time': timestamp, 'from': old_name, 'to': new_name}]}
        self.reid_events = {}  # {track_id: [{'time': timestamp, 'result': success/failure}]}
        
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
        
        # Add performance settings
        self.face_detection_interval = 5  # Only run face detection every N frames
        self.frame_count = 0
        self.last_face_detection_time = 0
        self.min_face_detection_interval = 0.5  # Minimum seconds between full face detections
        
        # Motion prediction parameters
        self.velocity_history = {}  # {id: [list of recent velocity vectors]}
        self.max_velocity_history = 3  # Keep last 3 velocity measurements
        self.use_motion_prediction = True  # Enable motion prediction for better tracking
        
        if model is None:
             print("WARNING: YOLO model not available. Face re-identification will be disabled.")

    def load_known_people(self):
        """Load known people with improved feature extraction"""
        if not os.path.exists(self.known_people_dir):
            print(f"Known people directory not found: {self.known_people_dir}")
            return

        for person_name in os.listdir(self.known_people_dir):
            person_dir = os.path.join(self.known_people_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            features = []
            images = []
            
            for img_file in os.listdir(person_dir):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                img_path = os.path.join(person_dir, img_file)
                try:
                    # Load and convert image
                    frame = cv2.imread(img_path)
                    if frame is None:
                        continue
                        
                    # Convert to RGB (face_recognition expects RGB)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect face locations using face_recognition (more reliable for stored images)
                    # face_locations = face_recognition.face_locations(rgb_frame)
                    #
                    # if not face_locations:
                    #     print(f"No face found in {img_path}")
                    #     continue
                    
                    # Get face encoding using face_recognition
                    # face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    face_encodings = face_recognition.face_encodings(rgb_frame)
                    
                    if face_encodings:
                        features.append(face_encodings[0])
                        images.append(img_path)
                        print(f"Successfully extracted features from {img_path}")
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

            if features:
                self.known_people[person_name] = {
                    'features': features,
                    'images': images
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
        """Extract face features using face_recognition library (based on dlib)"""
        try:
            # Bail early if frame is invalid
            if frame is None or frame.size == 0:
                return None
                
            # Convert bbox from (x1, y1, x2, y2) to dlib rectangle with padding
            x1, y1, x2, y2 = map(int, bbox)
            
            # Add padding to the bounding box (20% on each side)
            height = y2 - y1
            width = x2 - x1
            pad_x = int(width * 0.2)
            pad_y = int(height * 0.2)
            
            # Apply padding with boundary checks
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(frame.shape[1], x2 + pad_x)
            y2 = min(frame.shape[0], y2 + pad_y)
            
            # Skip tiny faces
            if (x2 - x1) < 30 or (y2 - y1) < 30:
                return None

            # Convert to RGB (face_recognition expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to face_recognition format (top, right, bottom, left)
            face_location = (y1, x2, y2, x1)
            
            # Extract face encoding using face_recognition (more reliable)
            # face_encodings = face_recognition.face_encodings(rgb_frame, [face_location]) # for some reason giving face_locations makes recognition worse
            face_encodings = face_recognition.face_encodings(rgb_frame)
            
            if not face_encodings:
                return None
                
            return face_encodings[0]
            
        except Exception as e:
            print(f"Error extracting face feature: {e}")
            return None

    # +++ Helper: Calculate Feature Distance +++
    def _calculate_feature_distance(self, feature1, feature2):
        """Calculate distance between two face features using selected distance metric"""
        if feature1 is None or feature2 is None:
            return float('inf')
        
        # Choose distance metric based on configuration
        if self.use_cosine_distance:
            # Cosine distance: 0 is identical, 1 is completely different
            distance = cosine(feature1, feature2)
        else:
            # Euclidean distance (face_recognition's default)
            # Using face_recognition's face_distance which is optimized for dlib encodings
            distance = face_recognition.face_distance([feature1], feature2)[0]
            
        return distance

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
        """Find best match with improved matching logic based on selected distance metric"""
        best_match_id = -1
        best_match_name = "UNK"
        min_distance = float('inf')
        
        metric_name = "Cosine" if self.use_cosine_distance else "Euclidean"
        threshold = self.similarity_threshold
        print(f"\n===== Face Matching ({metric_name}, threshold={threshold:.4f}) =====")
        
        # --- FIRST CHECK: Match with known people ---
        print(f"-- Checking against known people database --")
        # Check each known person individually
        for name, data in self.known_people.items():
            for i, known_feature in enumerate(data['features']):
                # Calculate distance using selected metric
                distance = self._calculate_feature_distance(known_feature, face_feature)
                print(f"Distance to {name} (feature {i+1}): {distance:.4f}")
                
                # Update best match if this is better
                if distance < min_distance and distance < self.similarity_threshold:
                    min_distance = distance
                    best_match_name = name
        
        # If we found a match with a known person
        if best_match_name != "UNK":
            print(f"Found match for known person {best_match_name} with distance {min_distance:.4f}")
            
            # Look for existing track with this name to maintain duration
            for track_id, track_data in self.tracked_objects.items():
                if track_data.get('name') == best_match_name and track_data.get('active', False):
                    best_match_id = track_id
                    break

        # --- SECOND CHECK: If no known person match, try recent tracks ---
        if best_match_id == -1 and best_match_name == "UNK":
            print(f"-- No match in known database, checking against recent tracks --")
            # Check each track individually
            for track_id, track_data in self.face_database.items():
                # Skip tracks that are too old
                if current_time - track_data['last_seen'] > self.reid_time_window:
                    continue
                
                # Get the best feature for this track
                track_feature = None
                if track_id in self.feature_history and len(self.feature_history[track_id]) > 0:
                    # Use average of recent features
                    track_feature = np.mean(self.feature_history[track_id], axis=0)
                elif 'feature' in track_data:
                    track_feature = track_data['feature']
                else:
                    continue
                
                # Calculate distance using selected metric
                distance = self._calculate_feature_distance(track_feature, face_feature)
                track_name = self.tracked_objects.get(track_id, {}).get('name', 'UNK')
                print(f"Distance to track #{track_id} (name: {track_name}): {distance:.4f}")
                
                # Update best match if this is better
                if distance < min_distance and distance < self.similarity_threshold:
                    min_distance = distance
                    best_match_id = track_id
                    best_match_name = self.tracked_objects.get(track_id, {}).get('name', 'UNK')
                    print(f"Re-identified track {best_match_id} with distance {min_distance:.4f}")
        
        print(f"===== Face Matching Result: ID={best_match_id}, Name={best_match_name}, Distance={min_distance:.4f} =====\n")
        return best_match_id, best_match_name, min_distance

    def _calculate_appearance_similarity(self, frame, bbox1, bbox2):
        """
        Calculate appearance similarity between two bounding boxes regions.
        Returns a similarity score between 0 and 1 (higher is more similar).
        """
        try:
            # Extract regions from frame
            x1_1, y1_1, x2_1, y2_1 = map(int, bbox1)
            x1_2, y1_2, x2_2, y2_2 = map(int, bbox2)
            
            # Ensure coordinates are within frame boundaries
            height, width = frame.shape[:2]
            x1_1, y1_1 = max(0, x1_1), max(0, y1_1)
            x2_1, y2_1 = min(width, x2_1), min(height, y2_1)
            x1_2, y1_2 = max(0, x1_2), max(0, y1_2)
            x2_2, y2_2 = min(width, x2_2), min(height, y2_2)
            
            # Skip if any region is too small
            if (x2_1 - x1_1) < 10 or (y2_1 - y1_1) < 10 or (x2_2 - x1_2) < 10 or (y2_2 - y1_2) < 10:
                return 0.0
                
            # Extract regions
            region1 = frame[y1_1:y2_1, x1_1:x2_1]
            region2 = frame[y1_2:y2_2, x1_2:x2_2]
            
            # Resize to same dimensions for comparison
            target_size = (64, 128)  # Common size for person appearance models
            region1_resized = cv2.resize(region1, target_size, interpolation=cv2.INTER_AREA)
            region2_resized = cv2.resize(region2, target_size, interpolation=cv2.INTER_AREA)
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(region1_resized, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(region2_resized, cv2.COLOR_BGR2GRAY)
            
            # Calculate histograms (fast appearance representation)
            hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
            
            # Normalize histograms
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            
            # Compare histograms
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # Handle NaN or invalid values
            if np.isnan(similarity):
                return 0.0
                
            # Ensure range [0, 1]
            return max(0.0, min(float(similarity), 1.0))
            
        except Exception as e:
            print(f"Error calculating appearance similarity: {e}")
            return 0.0

    def _hungarian_match_iou_appearance(self, frame, tracks, detections, current_time):
        """
        Enhanced Hungarian algorithm using both IoU and appearance similarity.
        
        Args:
            frame: The current video frame
            tracks: Dictionary of active tracks {id: track_data}
            detections: List of detection dictionaries with 'bbox' key
            current_time: Current timestamp
            
        Returns:
            matches: List of tuples (track_id, detection_idx)
            unmatched_tracks: List of track_ids that were not matched
            unmatched_detections: List of detection indices that were not matched
        """
        if not tracks or not detections:
            return [], list(tracks.keys()), list(range(len(detections)))
        
        # Create cost matrix (higher cost = less likely match)
        cost_matrix = np.ones((len(tracks), len(detections))) * float('inf')
        
        # Fill the cost matrix with a combination of IoU and appearance similarity
        for i, (track_id, track_data) in enumerate(tracks.items()):
            # Get current bbox
            track_bbox = track_data['bbox']
            
            # Apply motion prediction if enabled
            if self.use_motion_prediction and 'prev_time' in track_data:
                time_delta = current_time - track_data['prev_time']
                predicted_bbox = self._predict_bbox(track_id, time_delta)
            else:
                predicted_bbox = track_bbox
            
            for j, detection in enumerate(detections):
                det_bbox = detection['bbox']
                
                # Calculate IoU with both predicted and current bbox
                current_iou = self._calculate_iou(track_bbox, det_bbox)
                predicted_iou = self._calculate_iou(predicted_bbox, det_bbox)
                best_iou = max(current_iou, predicted_iou)
                
                # Only calculate appearance similarity if IoU is reasonable
                # This improves performance by skipping obviously poor matches
                if best_iou > 0.1:  # Lower threshold for appearance check
                    # Calculate appearance similarity
                    appearance_similarity = self._calculate_appearance_similarity(frame, track_bbox, det_bbox)
                    
                    # Combine IoU and appearance similarity (weighted average)
                    # Give more weight to IoU initially, but use appearance for resolving close cases
                    combined_score = 0.7 * best_iou + 0.3 * appearance_similarity
                    
                    # For boxes with significant overlap, give more weight to appearance
                    if best_iou > 0.5:
                        combined_score = 0.4 * best_iou + 0.6 * appearance_similarity
                        
                    # Only consider as match if combined score is good enough
                    if combined_score > self.iou_threshold:
                        # Convert to cost (lower is better for Hungarian algorithm)
                        cost_matrix[i, j] = 1.0 - combined_score
                    
        # Use Hungarian algorithm to find optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Create matches, unmatched_tracks, and unmatched_detections
        matches = []
        track_ids = list(tracks.keys())
        
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] != float('inf'):
                matches.append((track_ids[row], col))
        
        # Find unmatched tracks
        matched_track_indices = row_indices.tolist()
        unmatched_tracks = [track_id for i, track_id in enumerate(track_ids) 
                           if i not in matched_track_indices]
        
        # Find unmatched detections
        matched_detection_indices = col_indices.tolist()
        unmatched_detections = [j for j in range(len(detections)) 
                              if j not in matched_detection_indices]
        
        return matches, unmatched_tracks, unmatched_detections

    def _should_perform_reid(self, track_id, current_time):
        """Check if it's time to re-identify a track"""
        if not self.periodic_reid_enabled:
            return False
            
        # Skip for temporary tracks
        if isinstance(track_id, str):
            return False
            
        # Initialize last check time if not present
        if track_id not in self.last_reid_checks:
            self.last_reid_checks[track_id] = current_time - self.periodic_reid_interval + 1  # Check soon after creation
            return False
            
        # Check if enough time has passed since last check
        time_since_last_check = current_time - self.last_reid_checks[track_id]
        return time_since_last_check >= self.periodic_reid_interval

    def _handle_reid_result(self, track_id, face_feature, current_time):
        """Handle periodic re-identification result"""
        # Update last check time
        self.last_reid_checks[track_id] = current_time
        
        # Record the re-ID event
        if track_id not in self.reid_events:
            self.reid_events[track_id] = []
        
        # Add basic event record (will update with result)
        reid_event = {'time': current_time, 'result': 'no_face'}
        
        # Skip if face feature couldn't be extracted
        if face_feature is None:
            self.reid_events[track_id].append(reid_event)
            return False
            
        # Update event as face was extracted
        reid_event['result'] = 'face_extracted'
        
        # Get current track data
        if track_id not in self.tracked_objects:
            self.reid_events[track_id].append(reid_event)
            return False
            
        track_data = self.tracked_objects[track_id]
        current_name = track_data.get('name', 'UNK')
        
        print(f"\n===== Re-identification for Track #{track_id} (current name: {current_name}) =====")
        
        # Find best match for this face feature
        best_match_id, best_match_name, min_distance = self._find_best_face_match(face_feature, current_time)
        
        # Update re-ID event with match information
        reid_event['match_id'] = best_match_id
        reid_event['match_name'] = best_match_name
        reid_event['distance'] = min_distance
        
        # If we didn't find any match, just update feature history
        if best_match_id == -1 and best_match_name == "UNK":
            self._update_feature_history(track_id, face_feature)
            
            # Initialize failure count if needed
            if track_id not in self.reid_failure_counts:
                self.reid_failure_counts[track_id] = 0
                
            reid_event['result'] = 'no_match'
            self.reid_events[track_id].append(reid_event)
            return False
        
        # Check if the identified person matches current track
        identity_changed = False
        
        # Case 1: Known person, update to a known ID
        if best_match_name != "UNK" and current_name == "UNK":
            # We now know this person's identity
            print(f"Re-ID found identity for track {track_id}: {best_match_name}")
            
            # Record identity change
            if track_id not in self.identity_changes:
                self.identity_changes[track_id] = []
            
            self.identity_changes[track_id].append({
                'time': current_time,
                'from': current_name,
                'to': best_match_name,
                'reason': 'identified_unknown',
                'confidence': 1.0 - min_distance
            })
            
            # Update track data
            self.tracked_objects[track_id]['name'] = best_match_name
            identity_changed = True
            
            # Update known people tracks mapping
            if best_match_name not in self.known_people_tracks:
                self.known_people_tracks[best_match_name] = []
            if track_id not in self.known_people_tracks[best_match_name]:
                self.known_people_tracks[best_match_name].append(track_id)
            
            # Reset failure count
            self.reid_failure_counts[track_id] = 0
            
            # Update re-ID event
            reid_event['result'] = 'identified_unknown'
        
        # Case 2: Known person, different identity than we thought
        elif best_match_name != "UNK" and current_name != "UNK" and best_match_name != current_name:
            # Potential identity switch detected - increment failure count
            if track_id not in self.reid_failure_counts:
                self.reid_failure_counts[track_id] = 0
            
            self.reid_failure_counts[track_id] += 1
            
            # Update re-ID event with identity conflict
            reid_event['result'] = 'identity_conflict'
            reid_event['current_name'] = current_name
            reid_event['failure_count'] = self.reid_failure_counts[track_id]
            
            # Check if we've exceeded failure threshold
            if self.reid_failure_counts[track_id] >= self.reid_failure_threshold:
                print(f"Identity switch detected for track {track_id}: {current_name} -> {best_match_name}")
                
                # Record identity change
                if track_id not in self.identity_changes:
                    self.identity_changes[track_id] = []
                
                self.identity_changes[track_id].append({
                    'time': current_time,
                    'from': current_name,
                    'to': best_match_name,
                    'reason': 'identity_switch',
                    'confidence': 1.0 - min_distance,
                    'after_failures': self.reid_failure_counts[track_id]
                })
                
                # Remove from old known people track mapping
                if current_name in self.known_people_tracks and track_id in self.known_people_tracks[current_name]:
                    self.known_people_tracks[current_name].remove(track_id)
                
                # Update to new identity
                self.tracked_objects[track_id]['name'] = best_match_name
                
                # Add to new known people track mapping
                if best_match_name not in self.known_people_tracks:
                    self.known_people_tracks[best_match_name] = []
                if track_id not in self.known_people_tracks[best_match_name]:
                    self.known_people_tracks[best_match_name].append(track_id)
                
                identity_changed = True
                # Reset failure count
                self.reid_failure_counts[track_id] = 0
                
                # Update re-ID event for switch
                reid_event['result'] = 'identity_switched'
        
        # Case 3: Identity confirmed
        elif (best_match_name != "UNK" and current_name == best_match_name) or (best_match_id == track_id):
            # Identity confirmed - reset failure count
            self.reid_failure_counts[track_id] = 0
            
            # Update re-ID event
            reid_event['result'] = 'identity_confirmed'
        
        # Update feature history and database
        avg_feature = self._update_feature_history(track_id, face_feature)
        if track_id in self.face_database:
            self.face_database[track_id]['feature'] = avg_feature
            self.face_database[track_id]['last_seen'] = current_time
        
        # Save the re-ID event
        self.reid_events[track_id].append(reid_event)
        
        return identity_changed

    def update(self, frame, detections, current_time=None):
        """Update tracks with performance optimizations and Hungarian matching algorithm."""
        if current_time is None:
            current_time = time.time()
        
        # Track frame count for skipping face detection
        self.frame_count += 1
        should_detect_faces = (self.frame_count % self.face_detection_interval == 0) and \
                               (current_time - self.last_face_detection_time >= self.min_face_detection_interval)
        
        active_objects = {}
        matched_track_ids = set()
        matched_detection_indices = set()
        
        # --- STEP 1: Match active permanent tracks using enhanced Hungarian algorithm ---
        active_permanent_tracks = {id: data for id, data in self.tracked_objects.items() 
                                 if data.get('active', True) and not isinstance(id, str)}
        
        if active_permanent_tracks and detections:
            # Apply enhanced Hungarian algorithm that uses both IoU and appearance
            matches, unmatched_tracks, unmatched_detections = self._hungarian_match_iou_appearance(
                frame, active_permanent_tracks, detections, current_time
            )
            
            # Process matches
            for track_id, det_idx in matches:
                # Update track with matched detection
                bbox = detections[det_idx]['bbox']
                prev_bbox = self.tracked_objects[track_id]['bbox']
                
                # Calculate time delta for velocity update
                prev_time = self.tracked_objects[track_id].get('prev_time', current_time)
                dt = current_time - prev_time
                
                # Update track data
                self.tracked_objects[track_id]['bbox'] = bbox
                self.tracked_objects[track_id]['last_seen'] = current_time
                
                # Update velocity estimation
                if dt > 0:
                    velocity = self._update_velocity(track_id, bbox, dt)
                    if velocity:
                        self.tracked_objects[track_id]['velocity'] = velocity
                
                # Ensure time tracking continues
                if track_id in self.time_data:
                    self.time_data[track_id]['last_seen'] = current_time
                    intervals = self.time_data[track_id]['active_intervals']
                    if not intervals or intervals[-1][1] is not None:
                        intervals.append([current_time, None])
                
                # --- NEW: Periodic Re-identification Check ---
                if self._should_perform_reid(track_id, current_time):
                    # Extract face feature for re-identification
                    face_feature = self._extract_face_feature(frame, bbox)
                    self._handle_reid_result(track_id, face_feature, current_time)
                
                matched_track_ids.add(track_id)
                matched_detection_indices.add(det_idx)
            
            # Save unmatched detections for face detection step
            newly_detected_indices = set(unmatched_detections)
        else:
            # If no active tracks or detections, all detections are new
            newly_detected_indices = set(range(len(detections)))
        
        # --- STEP 2: Process remaining detections with face detection (only on some frames) ---
        if should_detect_faces:
            self.last_face_detection_time = current_time
            
            for det_idx in list(newly_detected_indices):
                if det_idx in matched_detection_indices:
                    continue  # Skip already matched detections
                    
                bbox = detections[det_idx]['bbox']
                face_feature = self._extract_face_feature(frame, bbox)
                
                if face_feature is not None:
                    print(f"\n===== New Face Detection (Detection #{det_idx}) =====")
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
                        matched_detection_indices.add(det_idx)
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
                        matched_detection_indices.add(det_idx)
        
        # Create temporary tracks for remaining detections (without face detection)
        matched_temp_ids = set()
        for det_idx in range(len(detections)):
            if det_idx in matched_detection_indices:
                continue  # Skip already matched detections
                
            bbox = detections[det_idx]['bbox']
            temp_id = f"temp_{self.temp_id_counter}"
            self.temp_id_counter += 1
            
            self.unidentified_tracks[temp_id] = {
                'bbox': bbox,
                'first_seen': current_time,
                'last_seen': current_time
            }
            matched_temp_ids.add(temp_id)
        
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
        
        # Update temporary tracks
        for temp_id in list(self.unidentified_tracks.keys()):
            if temp_id not in matched_temp_ids:
                if current_time - self.unidentified_tracks[temp_id]['last_seen'] > self.disappear_threshold:
                    # Remove temporary track if disappeared
                    del self.unidentified_tracks[temp_id]
        
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

    def _update_velocity(self, track_id, current_bbox, dt):
        """Update velocity estimation for a track"""
        if track_id not in self.tracked_objects:
            return None
            
        # Get previous bbox
        prev_bbox = self.tracked_objects[track_id].get('prev_bbox')
        if prev_bbox is None:
            # Store current bbox as previous for next update
            self.tracked_objects[track_id]['prev_bbox'] = current_bbox
            self.tracked_objects[track_id]['prev_time'] = time.time()
            return None
            
        # Calculate center points
        x1_prev, y1_prev, x2_prev, y2_prev = prev_bbox
        x1_curr, y1_curr, x2_curr, y2_curr = current_bbox
        
        center_prev_x = (x1_prev + x2_prev) / 2
        center_prev_y = (y1_prev + y2_prev) / 2
        center_curr_x = (x1_curr + x2_curr) / 2
        center_curr_y = (y1_curr + y2_curr) / 2
        
        # Calculate velocity (change in position over time)
        if dt > 0:
            velocity_x = (center_curr_x - center_prev_x) / dt
            velocity_y = (center_curr_y - center_prev_y) / dt
        else:
            velocity_x, velocity_y = 0, 0
            
        # Initialize velocity history if needed
        if track_id not in self.velocity_history:
            self.velocity_history[track_id] = []
            
        # Add current velocity to history
        self.velocity_history[track_id].append((velocity_x, velocity_y))
        
        # Keep only recent velocity measurements
        if len(self.velocity_history[track_id]) > self.max_velocity_history:
            self.velocity_history[track_id].pop(0)
            
        # Store current bbox as previous for next update
        self.tracked_objects[track_id]['prev_bbox'] = current_bbox
        self.tracked_objects[track_id]['prev_time'] = time.time()
        
        # Return average velocity
        if self.velocity_history[track_id]:
            avg_velocity_x = sum(v[0] for v in self.velocity_history[track_id]) / len(self.velocity_history[track_id])
            avg_velocity_y = sum(v[1] for v in self.velocity_history[track_id]) / len(self.velocity_history[track_id])
            return (avg_velocity_x, avg_velocity_y)
        return (0, 0)
        
    def _predict_bbox(self, track_id, time_delta):
        """Predict new bounding box position based on velocity"""
        if track_id not in self.tracked_objects or not self.use_motion_prediction:
            return self.tracked_objects[track_id]['bbox']
            
        # Get current bbox and velocity
        bbox = self.tracked_objects[track_id]['bbox']
        velocity = self.tracked_objects[track_id].get('velocity', (0, 0))
        
        if velocity is None or (velocity[0] == 0 and velocity[1] == 0):
            return bbox
            
        # Unpack values
        x1, y1, x2, y2 = bbox
        vel_x, vel_y = velocity
        
        # Predict new center based on velocity
        center_x = (x1 + x2) / 2 + vel_x * time_delta
        center_y = (y1 + y2) / 2 + vel_y * time_delta
        
        # Calculate width and height
        width = x2 - x1
        height = y2 - y1
        
        # Create new bbox with same dimensions
        new_x1 = center_x - width / 2
        new_y1 = center_y - height / 2
        new_x2 = center_x + width / 2
        new_y2 = center_y + height / 2
        
        return (new_x1, new_y1, new_x2, new_y2)

    def get_reid_stats(self):
        """Return statistics about re-identification and identity switches"""
        # Count total re-ID attempts
        reid_count = sum(len(events) for events in self.reid_events.values())
        
        # Count successful identity confirmations
        confirmed_count = 0
        for track_id, events in self.reid_events.items():
            for event in events:
                if event.get('result') == 'identity_confirmed':
                    confirmed_count += 1
        
        # Count identity switches
        switch_count = 0
        for track_id, changes in self.identity_changes.items():
            for change in changes:
                if change.get('reason') == 'identity_switch':
                    switch_count += 1
        
        # Count newly identified tracks (unknown to known)
        identified_count = 0
        for track_id, changes in self.identity_changes.items():
            for change in changes:
                if change.get('reason') == 'identified_unknown':
                    identified_count += 1
        
        return {
            'total_reid_attempts': reid_count,
            'identity_confirmations': confirmed_count,
            'identity_switches': switch_count,
            'newly_identified': identified_count
        }

    def set_distance_metric(self, use_cosine=False):
        """Change the distance metric used for feature comparison"""
        # If already using the requested metric, no change needed
        if self.use_cosine_distance == use_cosine:
            return
            
        # Swap thresholds before changing metric
        current_threshold = self.similarity_threshold
        self.similarity_threshold = self.cosine_similarity_threshold
        self.cosine_similarity_threshold = current_threshold
            
        # Switch the distance metric
        self.use_cosine_distance = use_cosine
        
        # Log the change
        metric_name = 'cosine' if use_cosine else 'Euclidean'
        print(f"Switched to {metric_name} distance (threshold: {self.similarity_threshold})")
            
        # Return the new threshold for confirmation
        return {
            'similarity_threshold': self.similarity_threshold,
            'metric': 'cosine' if self.use_cosine_distance else 'euclidean'
        }

        # Feature threshold settings
        feature_frame = ttk.Frame(self.reid_frame)
        feature_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(feature_frame, text="Similarity Threshold:").pack(side=tk.LEFT)
        
        self.similarity_threshold_var = tk.DoubleVar(value=0.6)
        self.similarity_threshold_spinbox = ttk.Spinbox(
            feature_frame,
            from_=0.1,
            to=1.0,
            increment=0.05,
            textvariable=self.similarity_threshold_var,
            width=5
        )
        self.similarity_threshold_spinbox.pack(side=tk.RIGHT)
        
        # Remove Re-ID threshold settings as we're using a single threshold

    def update_reid_settings(self):
        """Update Re-ID settings for all trackers"""
        # Get current values from UI
        enabled = self.reid_enabled_var.get()
        interval = self.reid_interval_var.get()
        switch_threshold = self.reid_threshold_var.get()
        similarity_threshold = self.similarity_threshold_var.get()
        use_cosine = self.distance_metric_var.get() == "cosine"
        
        # Update all trackers with new settings
        for camera_id, tracker in self.trackers.items():
            tracker.periodic_reid_enabled = enabled
            tracker.periodic_reid_interval = interval
            tracker.reid_failure_threshold = switch_threshold
            
            # Update threshold based on selected metric
            if use_cosine:
                tracker.cosine_similarity_threshold = similarity_threshold
            else:
                tracker.similarity_threshold = similarity_threshold
            
            # Set distance metric (this will apply the appropriate thresholds)
            tracker.set_distance_metric(use_cosine)
        
        self.log_message(
            f"Updated Re-ID settings: Enabled={enabled}, Interval={interval}s, "
            f"Switch Threshold={switch_threshold}, Similarity Threshold={similarity_threshold:.2f}, "
            f"Distance Metric={'Cosine' if use_cosine else 'Euclidean'}"
        )

class CameraManager:
    def __init__(self, camera_ids=None):
        self.cameras = {}  # {id: {'cap': VideoCapture, 'frame': frame, 'thread': thread, 'running': bool}}
        self.frame_queues = {}  # {id: Queue} - Raw frames from camera
        self.processed_frame_queues = {}  # {id: Queue} - Processed frames with detections
        self.frame_buffers = {}  # {id: deque} - Smooth display buffer
        self.max_queue_size = 5  # Maximum frames in processing queue (smaller to reduce latency)
        self.max_processed_queue_size = 2  # Only keep the most recent processed frames
        
        if camera_ids:
            for cam_id in camera_ids:
                self.add_camera(cam_id)
    
    def get_camera_ids(self):
        """Get list of camera IDs"""
        return list(self.cameras.keys())
    
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
        
        # Clear frame queues
        while not self.frame_queues[camera_id].empty():
            try:
                self.frame_queues[camera_id].get_nowait()
            except queue.Empty:
                break
        
        while not self.processed_frame_queues[camera_id].empty():
            try:
                self.processed_frame_queues[camera_id].get_nowait()
            except queue.Empty:
                break
        
        return True, f"Camera {camera_id} stopped"
    
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
        del self.processed_frame_queues[camera_id]
        del self.frame_buffers[camera_id]
        
        return True, f"Camera {camera_id} removed"
    
    def add_camera(self, camera_id):
        """Add a new camera to the manager"""
        try:
            # Convert to int if it's a numeric string
            if isinstance(camera_id, str) and camera_id.isdigit():
                camera_id = int(camera_id)
                
            # Create video capture with optimized settings
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                return False, f"Could not open camera {camera_id}"
            
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency
            cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30fps if camera supports it
            
            # Create frame queues and buffer
            self.frame_queues[camera_id] = queue.Queue(maxsize=self.max_queue_size)
            self.processed_frame_queues[camera_id] = queue.Queue(maxsize=self.max_processed_queue_size)
            self.frame_buffers[camera_id] = deque(maxlen=2)  # Only need latest frame for display
            
            # Store camera info
            self.cameras[camera_id] = {
                'cap': cap,
                'frame': None,
                'thread': None,
                'running': False,
                'last_capture_time': 0,
                'fps': 0
            }
            
            return True, f"Camera {camera_id} added successfully"
        except Exception as e:
            return False, f"Error adding camera {camera_id}: {str(e)}"
    
    def start_camera(self, camera_id):
        """Start capturing frames from a camera in a dedicated thread"""
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
    
    def _capture_frames(self, camera_id):
        """Dedicated thread for capturing frames from camera at maximum speed"""
        cap = self.cameras[camera_id]['cap']
        frame_queue = self.frame_queues[camera_id]
        
        frame_count = 0
        start_time = time.time()
        last_fps_update = start_time
        
        while self.cameras[camera_id]['running']:
            try:
                # Read frame as fast as possible
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.001)  # Tiny sleep to avoid CPU spin
                    continue
                
                current_time = time.time()
                frame_count += 1
                
                # Update FPS calculation every second
                if current_time - last_fps_update >= 1.0:
                    self.cameras[camera_id]['fps'] = frame_count / (current_time - start_time)
                    frame_count = 0
                    start_time = current_time
                    last_fps_update = current_time
                
                # Record last successful capture time
                self.cameras[camera_id]['last_capture_time'] = current_time
                
                # Always replace oldest frame if queue is full
                try:
                    # Try non-blocking put first
                    frame_queue.put_nowait((frame, current_time))
                except queue.Full:
                    # If full, remove oldest and then add new frame
                    try:
                        frame_queue.get_nowait()
                        frame_queue.put_nowait((frame, current_time))
                    except (queue.Empty, queue.Full):
                        # In case of race condition, just continue
                        pass
                
            except Exception as e:
                print(f"Error capturing frame from camera {camera_id}: {e}")
                time.sleep(0.01)  # Short sleep on error
        
        print(f"Camera {camera_id} capture thread stopped")

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
        
        # Add Re-ID settings
        self.reid_frame = ttk.LabelFrame(settings_frame, text="Re-ID Settings")
        self.reid_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Enable Re-ID
        self.reid_enabled_var = tk.BooleanVar(value=True)
        self.enable_reid_check = ttk.Checkbutton(
            self.reid_frame, 
            text="Enable Periodic Re-Identification", 
            variable=self.reid_enabled_var
        )
        self.enable_reid_check.pack(pady=5, fill=tk.X)
        
        # Re-ID Interval (seconds)
        interval_frame = ttk.Frame(self.reid_frame)
        interval_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(interval_frame, text="Re-ID Interval (seconds):").pack(side=tk.LEFT)
        
        self.reid_interval_var = tk.DoubleVar(value=5.0)
        self.reid_interval_spinbox = ttk.Spinbox(
            interval_frame, 
            from_=1.0, 
            to=30.0, 
            increment=0.5,
            textvariable=self.reid_interval_var,
            width=5
        )
        self.reid_interval_spinbox.pack(side=tk.RIGHT)
        
        # Identity Switch Threshold
        threshold_frame = ttk.Frame(self.reid_frame)
        threshold_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(threshold_frame, text="Identity Switch Threshold:").pack(side=tk.LEFT)
        
        self.reid_threshold_var = tk.IntVar(value=3)
        self.reid_threshold_spinbox = ttk.Spinbox(
            threshold_frame, 
            from_=1, 
            to=10, 
            increment=1,
            textvariable=self.reid_threshold_var,
            width=5
        )
        self.reid_threshold_spinbox.pack(side=tk.RIGHT)
        
        # Similarity threshold
        similarity_frame = ttk.Frame(self.reid_frame)
        similarity_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(similarity_frame, text="Similarity Threshold:").pack(side=tk.LEFT)
        
        self.similarity_threshold_var = tk.DoubleVar(value=0.6)
        self.similarity_threshold_spinbox = ttk.Spinbox(
            similarity_frame,
            from_=0.1,
            to=1.0,
            increment=0.05,
            textvariable=self.similarity_threshold_var,
            width=5
        )
        self.similarity_threshold_spinbox.pack(side=tk.RIGHT)
        
        # Distance Metric selection
        ttk.Label(self.reid_frame, text="Distance Metric:").pack(anchor='w', padx=5, pady=(5, 0))
        self.distance_metric_var = tk.StringVar(value="euclidean")
        self.distance_metric_frame = ttk.Frame(self.reid_frame)
        self.distance_metric_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.euclidean_radio = ttk.Radiobutton(
            self.distance_metric_frame,
            text="Euclidean",
            variable=self.distance_metric_var,
            value="euclidean"
        )
        self.euclidean_radio.pack(side=tk.LEFT, padx=(0, 10))
        
        self.cosine_radio = ttk.Radiobutton(
            self.distance_metric_frame,
            text="Cosine",
            variable=self.distance_metric_var,
            value="cosine"
        )
        self.cosine_radio.pack(side=tk.LEFT)
        
        # Apply Settings button
        self.reid_apply_btn = ttk.Button(
            self.reid_frame, 
            text="Apply Settings", 
            command=self.update_reid_settings
        )
        self.reid_apply_btn.pack(pady=5, fill=tk.X)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(self.right_frame, text="Statistics")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.people_count_var = tk.StringVar(value="People count: 0")
        ttk.Label(stats_frame, textvariable=self.people_count_var).pack(anchor=tk.W, padx=5, pady=5)
        
        self.fps_var = tk.StringVar(value="FPS: 0")
        ttk.Label(stats_frame, textvariable=self.fps_var).pack(anchor=tk.W, padx=5, pady=5)
        
        # Re-ID statistics
        self.reid_stats_var = tk.StringVar(value="Re-IDs: 0 | ID Switches: 0")
        ttk.Label(stats_frame, textvariable=self.reid_stats_var).pack(anchor=tk.W, padx=5, pady=5)
        
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
        
        # Re-ID statistics tracking
        self.reid_counter = 0
        self.identity_switch_counter = 0
        
        # Improved thread control and monitoring
        self.processing_thread = None
        self.display_thread = None
        self.is_tracking = False
        self.processing_active = False
        self.display_active = False
        
        # Control rates for different operations
        self.frame_process_delay = 0.033  # ~30 FPS target for processing
        self.display_update_ms = 50  # ~20 FPS target for display
        
        # Performance monitoring
        self.processing_fps = 0
        self.display_fps = 0
        self.processing_frame_count = 0
        self.display_frame_count = 0
        self.process_start_time = 0
        self.display_start_time = 0
        
        # Protocol for window closing
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.log_message("Application started. Add cameras and click 'Start Tracking'.")
        self.log_message(f"Using device: {device}")
        
        # Add default camera
        self.add_camera() # This calls setup_camera_grid

    def update_reid_settings(self):
        """Update Re-ID settings for all trackers"""
        # Get current values from UI
        enabled = self.reid_enabled_var.get()
        interval = self.reid_interval_var.get()
        switch_threshold = self.reid_threshold_var.get()
        similarity_threshold = self.similarity_threshold_var.get()
        use_cosine = self.distance_metric_var.get() == "cosine"
        
        # Update all trackers with new settings
        for camera_id, tracker in self.trackers.items():
            tracker.periodic_reid_enabled = enabled
            tracker.periodic_reid_interval = interval
            tracker.reid_failure_threshold = switch_threshold
            
            # Update threshold based on selected metric
            if use_cosine:
                tracker.cosine_similarity_threshold = similarity_threshold
            else:
                tracker.similarity_threshold = similarity_threshold
            
            # Set distance metric (this will apply the appropriate thresholds)
            tracker.set_distance_metric(use_cosine)
        
        self.log_message(
            f"Updated Re-ID settings: Enabled={enabled}, Interval={interval}s, "
            f"Switch Threshold={switch_threshold}, Similarity Threshold={similarity_threshold:.2f}, "
            f"Distance Metric={'Cosine' if use_cosine else 'Euclidean'}"
        )

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
        
        # Create a tracker for this camera with current Re-ID settings
        tracker = PersonTracker()
        tracker.periodic_reid_enabled = self.reid_enabled_var.get()
        tracker.periodic_reid_interval = self.reid_interval_var.get()
        tracker.reid_failure_threshold = self.reid_threshold_var.get()
        self.trackers[camera_id] = tracker
        
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
    
    def process_frames_loop(self):
        """Dedicated thread that processes frames at a controlled rate"""
        self.log_message("Processing thread started")
        frame_count = 0
        start_time = time.time()
        last_fps_update = start_time
        camera_ids = self.camera_manager.get_camera_ids()
        
        # Create a processing queue to handle one camera at a time
        camera_queue = deque(camera_ids)
        
        # Using separate detection and tracking batches for better parallelism
        detection_batch = {}  # Store frames for object detection
        tracking_results = {}  # Store tracking results
        
        # Re-ID event counters
        reid_count = 0
        id_switch_count = 0
        id_confirm_count = 0
        id_new_count = 0
        last_stats_update = start_time
        
        while self.processing_active:
            process_start = time.time()
            processed_this_loop = False
            active_people_counts = {}
            
            if not camera_queue:
                camera_queue.extend(camera_ids)  # Refill the queue if empty
            
            # Process one camera per iteration for more balanced processing
            if camera_queue:
                camera_id = camera_queue[0]
                camera_queue.popleft()  # Remove the camera we're about to process
                
                try:
                    # Try to get the latest frame from the queue
                    try:
                        frame, timestamp = self.camera_manager.frame_queues[camera_id].get_nowait()
                    except queue.Empty:
                        time.sleep(0.001)  # Tiny sleep to avoid CPU spin
                        continue

                    processed_this_loop = True
                    
                    # Store frame for detection
                    detection_batch[camera_id] = frame
                    
                    # STEP 1: Run object detection (expensive but needed for all frames)
                    detections = self.detect_people(frame)
                    
                    # STEP 2: Update tracker (IoU + occasional Face Re-ID)
                    tracker = self.trackers[camera_id]
                    
                    # Capture Re-ID stats before update
                    pre_reid_stats = tracker.get_reid_stats()
                    
                    tracked_objects = tracker.update(frame, detections)
                    
                    # Capture Re-ID stats after update and calculate differences
                    post_reid_stats = tracker.get_reid_stats()
                    
                    # Calculate changes in stats
                    new_reids = post_reid_stats['total_reid_attempts'] - pre_reid_stats['total_reid_attempts']
                    new_switches = post_reid_stats['identity_switches'] - pre_reid_stats['identity_switches']
                    new_confirmations = post_reid_stats['identity_confirmations'] - pre_reid_stats['identity_confirmations']
                    new_identifications = post_reid_stats['newly_identified'] - pre_reid_stats['newly_identified']
                    
                    # Accumulate counters
                    reid_count += new_reids
                    id_switch_count += new_switches
                    id_confirm_count += new_confirmations
                    id_new_count += new_identifications
                    
                    # Log significant events
                    if new_reids > 0:
                        self.log_message(f"Performed {new_reids} re-ID(s) on camera {camera_id}")
                    
                    if new_switches > 0:
                        self.log_message(f"Detected {new_switches} identity switch(es) on camera {camera_id}")
                        
                    if new_identifications > 0:
                        self.log_message(f"Newly identified {new_identifications} person(s) on camera {camera_id}")
                    
                    # Store tracking results
                    tracking_results[camera_id] = tracked_objects
                    
                    # STEP 3: Draw results (on a copy)
                    result_frame = self.draw_results(frame.copy(), tracked_objects)
                    
                    # Track statistics
                    active_count = sum(1 for obj in tracked_objects.values() if obj.get('active', False))
                    active_people_counts[camera_id] = active_count
                    frame_count += 1
                    self.processing_frame_count += 1
                    
                    # Store processed frame result - always replace older frames
                    try:
                        # Non-blocking put
                        self.camera_manager.processed_frame_queues[camera_id].put_nowait(
                            (result_frame, tracked_objects)
                        )
                    except queue.Full:
                        # If full, clear and add new frame
                        try:
                            self.camera_manager.processed_frame_queues[camera_id].get_nowait()
                            self.camera_manager.processed_frame_queues[camera_id].put_nowait(
                                (result_frame, tracked_objects)
                            )
                        except (queue.Empty, queue.Full):
                            pass
                    
                except Exception as e:
                    self.log_message(f"Error processing frame from camera {camera_id}: {str(e)}")

            # Calculate processing rate
            current_time = time.time()
            if current_time - last_fps_update >= 1.0:
                # Update processing FPS counter
                elapsed = current_time - start_time
                if elapsed > 0:
                    self.processing_fps = frame_count / elapsed
                    
                # Reset counters
                frame_count = 0
                start_time = current_time
                last_fps_update = current_time
                
                # Update total active people count
                total_people = sum(active_people_counts.values())
                
                # Schedule UI update on main thread
                self.window.after(0, lambda: self.people_count_var.set(f"People count: {total_people}"))
                
                # Update Re-ID statistics (every second)
                self.reid_counter += reid_count
                self.identity_switch_counter += id_switch_count
                
                # Schedule detailed Re-ID stats update
                self.window.after(0, lambda: self.reid_stats_var.set(
                    f"Re-IDs: {self.reid_counter} | Switches: {self.identity_switch_counter} | New IDs: {id_new_count}"
                ))
                
                # Reset the counters for next update
                reid_count = 0
                id_switch_count = 0
                id_confirm_count = 0
                id_new_count = 0
            
            # Control the processing rate - sleep only if we processed a frame
            if processed_this_loop:
                process_time = time.time() - process_start
                sleep_time = max(0, self.frame_process_delay - process_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                # If no frames were processed, add a tiny sleep to avoid CPU spin
                time.sleep(0.001)
        
        self.log_message("Processing thread stopped")
    
    def _get_reid_stats(self, tracker):
        """Get the reid and identity switch statistics from a tracker"""
        # Use the tracker's method directly
        return tracker.get_reid_stats()
    
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
        self.processing_active = True
        self.display_active = True
        
        # Reset performance counters
        self.processing_frame_count = 0
        self.display_frame_count = 0
        self.process_start_time = time.time()
        self.display_start_time = time.time()
        
        # Update UI state
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_add_camera.config(state=tk.DISABLED)
        self.btn_remove_camera.config(state=tk.DISABLED)
        
        # Start processing thread (runs independently)
        self.processing_thread = threading.Thread(
            target=self.process_frames_loop, 
            daemon=True, 
            name="ProcessingThread"
        )
        self.processing_thread.start()
        
        # Start display thread (runs independently)
        self.display_thread = threading.Thread(
            target=self.display_frames_loop,
            daemon=True,
            name="DisplayThread"
        )
        self.display_thread.start()
        
        # Start stats update loop
        self.update_stats_loop()

        self.log_message(f"Tracking started with {len(camera_ids)} cameras.")
    
    def stop_tracking(self):
        if not self.is_tracking:
            return

        # Signal all threads to stop
        self.is_tracking = False
        self.processing_active = False
        self.display_active = False

        # Wait for processing thread to finish (add a timeout)
        if self.processing_thread and self.processing_thread.is_alive():
             self.log_message("Waiting for processing thread to finish...")
             self.processing_thread.join(timeout=1.0)
             if self.processing_thread.is_alive():
                 self.log_message("Warning: Processing thread did not exit cleanly.")
                 
        # Wait for display thread to finish
        if self.display_thread and self.display_thread.is_alive():
             self.log_message("Waiting for display thread to finish...")
             self.display_thread.join(timeout=1.0)
             if self.display_thread.is_alive():
                 self.log_message("Warning: Display thread did not exit cleanly.")

        # Stop all cameras (releases resources)
        self.log_message("Stopping cameras...")
        self.camera_manager.stop_all_cameras()
        
        # Update UI state
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_add_camera.config(state=tk.NORMAL)
        self.btn_remove_camera.config(state=tk.NORMAL)

        # Reset stats
        self.fps_var.set("FPS: 0")
        self.people_count_var.set("People count: 0")

        self.log_message("Tracking stopped.")

    def display_frames_loop(self):
        """Dedicated thread that updates the GUI at a controlled rate"""
        self.log_message("Display thread started")
        frame_count = 0
        start_time = time.time()
        last_fps_update = start_time
        
        while self.display_active:
            display_start = time.time()
            displayed_this_loop = False
            
            for camera_id in self.camera_manager.get_camera_ids():
                # Skip if no canvas exists for this camera
                if camera_id not in self.camera_canvases:
                    continue
                    
                try:
                    # Try to get the latest processed frame
                    try:
                        result_frame, _ = self.camera_manager.processed_frame_queues[camera_id].get_nowait()
                        
                        # Convert and prepare for display (in this thread)
                        display_image = self.prepare_display_image(camera_id, result_frame)
                        
                        # Schedule the actual canvas update on the main thread
                        self.window.after(0, lambda cam=camera_id, img=display_image: 
                                          self.update_canvas(cam, img))
                        
                        displayed_this_loop = True
                        frame_count += 1
                        self.display_frame_count += 1
                        
                    except queue.Empty:
                        # No new frames to display
                        pass
                    
                except Exception as e:
                    self.log_message(f"Error displaying frame from camera {camera_id}: {str(e)}")
            
            # Calculate display rate
            current_time = time.time()
            if current_time - last_fps_update >= 1.0:
                # Update display FPS counter
                elapsed = current_time - start_time
                if elapsed > 0:
                    self.display_fps = frame_count / elapsed
                
                # Reset counters
                frame_count = 0
                start_time = current_time
                last_fps_update = current_time
            
            # Control the display rate - sleep only if we displayed a frame
            if displayed_this_loop:
                display_time = time.time() - display_start
                sleep_time = max(0, (self.display_update_ms / 1000.0) - display_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                # If no frames were displayed, add a tiny sleep to avoid CPU spin
                time.sleep(0.01)
        
        self.log_message("Display thread stopped")
    
    def prepare_display_image(self, camera_id, frame):
        """Prepares a frame for display by converting it to Tkinter PhotoImage"""
        canvas = self.camera_canvases[camera_id]
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # If canvas dimensions are invalid, use default size
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 320
            canvas_height = 240
        
        # Calculate aspect ratio preserving resize
        frame_height, frame_width = frame.shape[:2]
        if frame_width == 0 or frame_height == 0:
            return None  # Skip empty frame
            
        scale = min(canvas_width / frame_width, canvas_height / frame_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        
        # Resize frame efficiently
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB for PIL -> Tkinter
        img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        photo = ImageTk.PhotoImage(image=img_pil)
        
        return photo
    
    def update_canvas(self, camera_id, photo):
        """Updates a canvas with the new image (called on main thread)"""
        if not self.is_tracking or camera_id not in self.camera_canvases:
            return
            
        canvas = self.camera_canvases[camera_id]
        
        # Store reference to prevent garbage collection
        self.camera_photos[camera_id] = photo
        
        # Find existing image item and update it
        img_item = canvas.find_withtag("frame_img")
        if img_item:
            canvas.itemconfig(img_item, image=photo)
        else:
            # Create new image if none exists
            canvas_width = canvas.winfo_width() or 320
            canvas_height = canvas.winfo_height() or 240
            canvas.create_image(
                canvas_width // 2, 
                canvas_height // 2, 
                image=photo, 
                anchor=tk.CENTER, 
                tags="frame_img"
            )
    
    def update_stats_loop(self):
        """Updates performance statistics in the UI"""
        if not self.is_tracking:
            return
            
        # Update FPS display with both processing and display rates
        fps_text = f"Processing: {self.processing_fps:.1f} FPS | Display: {self.display_fps:.1f} FPS"
        
        # Add camera capture FPS if available
        camera_fps = []
        for camera_id, camera_data in self.camera_manager.cameras.items():
            if 'fps' in camera_data:
                camera_fps.append(f"Cam {camera_id}: {camera_data['fps']:.1f}")
        
        if camera_fps:
            camera_fps_text = " | ".join(camera_fps)
            # Include camera FPS in the main display
            fps_text += f" | {camera_fps_text}"
        
        # Update the FPS display
        self.fps_var.set(fps_text)
        
        # Also log the FPS data periodically (every 5 seconds) to avoid log spam
        current_time = time.time()
        if not hasattr(self, "_last_fps_log") or current_time - self._last_fps_log >= 5.0:
            self.log_message(f"FPS Stats: {fps_text}")
            self._last_fps_log = current_time
        
        # Schedule next update (100ms for more responsive UI)
        self.window.after(100, self.update_stats_loop)
    
    def detect_people(self, frame):
        """Detect people in a frame using YOLOv8"""
        detections = []
        if model is None: return detections
        
        try:
            # Ensure frame is in expected format (e.g., BGR numpy array)
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
        """Draw bounding boxes and labels with duration for tracked objects"""
        result = frame.copy()
        current_time = time.time()
        
        # Draw all tracked objects
        for obj_id, obj_data in tracked_objects.items():
            if not obj_data.get('active', True):
                continue # Don't draw inactive tracks

            x1, y1, x2, y2 = map(int, obj_data['bbox'])
            
            # Check if this is a temporary track (awaiting face detection)
            is_temporary = obj_data.get('is_temporary', False)
            name = obj_data.get('name', 'UNK')
            
            # Check if this track was recently re-identified
            recently_reid = False
            reid_time = None
            
            # For permanent tracks, check if a re-ID happened in the last second
            if not is_temporary and not isinstance(obj_id, str):
                for tracker in self.trackers.values():
                    if obj_id in tracker.last_reid_checks:
                        reid_time = tracker.last_reid_checks[obj_id]
                        recently_reid = (current_time - reid_time) < 1.0  # Show indicator for 1 second
                        break
            
            # Calculate duration
            if is_temporary:
                if 'first_seen' in obj_data:
                    duration = current_time - obj_data['first_seen']
                    duration_str = self._format_duration(duration)
                else:
                    duration_str = "00:00:00"
                    
                # Yellow color for temporary tracks
                color = (0, 255, 255)  # Yellow in BGR
                label = f"Waiting... ({duration_str})"
            else:
                # For regular tracks, find the time data in tracker
                if isinstance(obj_id, str):
                    # This is a temp ID with no time data yet
                    duration_str = "00:00:00"
                else:
                    # For regular tracks, check all cameras for time data
                    camera_ids = list(self.trackers.keys())
                    for camera_id in camera_ids:
                        if obj_id in self.trackers[camera_id].time_data:
                            time_data = self.trackers[camera_id].time_data[obj_id]
                            # Calculate current duration from intervals
                            total_duration = time_data.get('total_active_time', 0)
                            
                            # Add duration of current active interval if exists
                            intervals = time_data.get('active_intervals', [])
                            if intervals and intervals[-1][1] is None:
                                current_interval = current_time - intervals[-1][0]
                                total_duration += current_interval
                                
                            duration_str = self._format_duration(total_duration)
                            break
                    else:
                        duration_str = "00:00:00"
                
                # Regular track color based on identification
                if name == "UNK":
                    # Orange for unknown but detected faces
                    color = (0, 165, 255)  # Orange in BGR
                    label = f"#{obj_id} ({duration_str})"
                else:
                    # Green for known people
                    color = (0, 255, 0)  # Green in BGR
                    label = f"{name} ({duration_str})"
                
                # Change color if recently re-identified
                if recently_reid:
                    # Bright cyan for recently re-identified tracks
                    color = (255, 255, 0)  # Cyan in BGR
                    
                    # Add "Re-ID" to the label
                    if reid_time:
                        time_since_reid = current_time - reid_time
                        label = f"{label} [Re-ID: {time_since_reid:.1f}s ago]"
                    else:
                        label = f"{label} [Re-ID]"
                    
                    # Check if this track has had identity switches
                    for tracker in self.trackers.values():
                        if obj_id in tracker.reid_failure_counts and tracker.reid_failure_counts[obj_id] > 0:
                            # Add warning for potential identity confusion
                            label += f" [Switch Confidence: {tracker.reid_failure_counts[obj_id]}/{tracker.reid_failure_threshold}]"
                            break

            # Draw box with thickness proportional to box size
            thickness = max(1, min(3, int((x2-x1) / 200 + 1)))
            
            # Draw the bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Calculate label position (above the box)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(result, 
                         (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0] + 10, y1), 
                         color, -1)  # Filled background
            
            # Draw text with black color for better contrast
            cv2.putText(result, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                       
            # Draw indicator on top of the box for recently re-identified
            if recently_reid:
                # Draw a small circle on top of the box
                center_x = x1 + (x2 - x1) // 2
                cv2.circle(result, (center_x, y1 - 20), 8, (255, 255, 0), -1)  # Filled cyan circle
                
                # Draw Re-ID text inside circle
                cv2.putText(result, "R", (center_x - 4, y1 - 17), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
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
                "unknown_people": [],
                "identity_changes": {},  # New section for tracking identity changes
                "reid_events": {}       # New section for re-ID events
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
                        
                        # Add identity changes for this track if any
                        if track_id in tracker.identity_changes and tracker.identity_changes[track_id]:
                            changes = tracker.identity_changes[track_id]
                            for change in changes:
                                change_copy = change.copy()
                                change_copy['time'] = datetime.fromtimestamp(change['time']).strftime("%Y-%m-%d %H:%M:%S")
                                
                                # Add to identity changes
                                if str(track_id) not in structured_data["identity_changes"]:
                                    structured_data["identity_changes"][str(track_id)] = []
                                
                                change_copy['camera'] = camera_id
                                structured_data["identity_changes"][str(track_id)].append(change_copy)
                        
                        # Add re-ID events for this track if any
                        if track_id in tracker.reid_events and tracker.reid_events[track_id]:
                            events = tracker.reid_events[track_id]
                            for event in events:
                                event_copy = event.copy()
                                event_copy['time'] = datetime.fromtimestamp(event['time']).strftime("%Y-%m-%d %H:%M:%S")
                                
                                # Add to re-ID events
                                if str(track_id) not in structured_data["reid_events"]:
                                    structured_data["reid_events"][str(track_id)] = []
                                
                                event_copy['camera'] = camera_id
                                structured_data["reid_events"][str(track_id)].append(event_copy)
                
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
                        
                        # Add identity changes for unknown tracks too
                        if track_id in tracker.identity_changes and tracker.identity_changes[track_id]:
                            changes = tracker.identity_changes[track_id]
                            for change in changes:
                                change_copy = change.copy()
                                change_copy['time'] = datetime.fromtimestamp(change['time']).strftime("%Y-%m-%d %H:%M:%S")
                                
                                # Add to identity changes
                                if str(track_id) not in structured_data["identity_changes"]:
                                    structured_data["identity_changes"][str(track_id)] = []
                                
                                change_copy['camera'] = camera_id
                                structured_data["identity_changes"][str(track_id)].append(change_copy)
            
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
            
            # Log identity changes
            if structured_data["identity_changes"]:
                self.log_message("\nIdentity Changes:")
                for track_id, changes in structured_data["identity_changes"].items():
                    self.log_message(f"  Track {track_id}:")
                    for change in changes:
                        self.log_message(f"    {change['time']}: {change['from']} → {change['to']} ({change['reason']})")
            
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
        traceback.print_exc()
