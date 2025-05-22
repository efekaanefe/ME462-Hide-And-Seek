import cv2
import numpy as np
import torch
from tkinter import ttk, messagebox, filedialog
import time
from datetime import datetime
import csv
import queue
from collections import deque
from scipy.spatial.distance import cosine # For feature comparison
import os
import json
from ultralytics import YOLO
import insightface  # Using ArcFace instead of dlib
from scipy.optimize import linear_sum_assignment  # Added for Hungarian Algorithm

# Suppress unnecessary warnings and logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["INSIGHTFACE_LOG_LEVEL"] = "ERROR"

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLOv8 model - using Ultralytics implementation
try:
    model = YOLO('models/yolov8n.pt')  # Load the smallest YOLOv8 model to start with
except ImportError:
    messagebox.showerror("Error", "Please install ultralytics: pip install ultralytics")
    model = None

# Load ArcFace model
try:
    face_model = insightface.app.FaceAnalysis(name='buffalo_l')
    face_model.prepare(ctx_id=0 if device == 'cuda' else -1)  # Use GPU if available
except ImportError:
    messagebox.showerror("Error", "Please install insightface: pip install insightface-python onnxruntime-gpu")
    face_model = None


class PersonTracker:
    def __init__(self):
        self.next_id = 1
        self.tracked_objects = {}  # Active/inactive tracks
        self.face_database = {}  # Permanent storage: {id: {'feature': feature_vector, 'last_seen': timestamp}}
        self.disappear_threshold = 2.0
        self.reid_time_window = 100.0  # Very long to effectively keep all tracks for re-id
        # self.iou_threshold = 0.7  # Minimum IoU for matching consideration
        
        # Set distance metric for feature comparison
        self.use_cosine_distance = True  # Default to cosine for ArcFace
        
        # Single similarity threshold (used for both feature matching and re-identification)
        self.similarity_threshold = 0.85  # Default threshold for Euclidean distance with ArcFace
        self.cosine_similarity_threshold = 0.55  # Default threshold for ArcFace cosine distance
        
        # Use the appropriate threshold based on the distance metric
        if self.use_cosine_distance:
            self.similarity_threshold = self.cosine_similarity_threshold
        
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
        """Load known people with ArcFace feature extraction"""
        if not os.path.exists(self.known_people_dir):
            print(f"Known people directory not found: {self.known_people_dir}")
            return
            
        if face_model is None:
            print("ArcFace model not available. Cannot load known people.")
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
                    # Load image
                    frame = cv2.imread(img_path)
                    if frame is None:
                        continue
                        
                    # Detect faces using ArcFace
                    faces = face_model.get(frame)
                    
                    if not faces:
                        print(f"No face found in {img_path}")
                        continue
                    
                    # Use the face with highest detection score
                    best_face = max(faces, key=lambda x: x.det_score) if len(faces) > 1 else faces[0]
                    
                    # Get and normalize the embedding
                    embedding = best_face.embedding
                    normalized_embedding = embedding / np.linalg.norm(embedding)
                    
                    features.append(normalized_embedding)
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
        """Add new images for a person to the dataset using ArcFace."""
        if face_model is None:
            print("ArcFace model not available. Cannot add person images.")
            return 0
            
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
                
                # Extract face using ArcFace
                faces = face_model.get(img)
                
                if not faces:
                    print(f"Warning: No face detected in {img_path}")
                    continue
                    
                # Use the face with highest detection score
                best_face = max(faces, key=lambda x: x.det_score) if len(faces) > 1 else faces[0]
                
                # Get and normalize the embedding
                embedding = best_face.embedding
                normalized_embedding = embedding / np.linalg.norm(embedding)
                
                # Copy image to person's directory
                new_path = os.path.join(person_dir, os.path.basename(img_path))
                cv2.imwrite(new_path, img)
                
                features.append(normalized_embedding)
                saved_paths.append(new_path)
                
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
        """Extract face features using ArcFace model"""
        try:
            # Bail early if frame is invalid or face_model not available
            if frame is None or frame.size == 0 or face_model is None:
                return None
                
            # Convert bbox from (x1, y1, x2, y2) to integers with padding
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

            # Create a cropped image for face detection
            face_img = frame[y1:y2, x1:x2]
            
            # Get face embedding using ArcFace
            faces = face_model.get(face_img)
            
            if not faces:
                # Try with the full frame if crop fails
                faces = face_model.get(frame)
                
                # Filter faces that overlap with our bbox
                if faces:
                    valid_faces = []
                    for face in faces:
                        face_box = face.bbox.astype(int)
                        # Check if the face overlaps with our person bbox
                        if self._calculate_iou([face_box[0], face_box[1], face_box[2], face_box[3]], 
                                              [x1, y1, x2, y2]) > 0.5:
                            valid_faces.append(face)
                    faces = valid_faces
            
            if not faces:
                return None
                
            # Get the embedding for the face with highest confidence
            best_face = max(faces, key=lambda x: x.det_score) if len(faces) > 1 else faces[0]
            embedding = best_face.embedding
            
            # Normalize the embedding (important for ArcFace comparison)
            normalized_embedding = embedding / np.linalg.norm(embedding)
            
            return normalized_embedding
            
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
            # For ArcFace, cosine similarity is preferred
            # Cosine similarity: 1 is identical, -1 is completely different
            # Convert to distance: 0 is identical, 2 is completely different
            similarity = np.dot(feature1, feature2)
            # Return 1-similarity so lower is better (consistent with other distance metrics)
            return 1 - similarity
        else:
            # Euclidean distance
            return np.linalg.norm(feature1 - feature2)

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

    def _hungarian_match_iou(self, frame, tracks, detections, current_time):
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

                cost_matrix[i, j] = -best_iou

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
            matches, unmatched_tracks, unmatched_detections = self._hungarian_match_iou(
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
            
        # For ArcFace, we need different thresholds
        if use_cosine:
            # ArcFace with cosine similarity (1-similarity becomes distance)
            self.cosine_similarity_threshold = 0.36  # Lower means more strict
        else:
            # Euclidean distance for ArcFace
            self.similarity_threshold = 0.85  # Higher means more strict
            
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