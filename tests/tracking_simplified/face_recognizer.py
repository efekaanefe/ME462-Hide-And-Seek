import cv2
import numpy as np
import insightface
import os
import time
from typing import List, Dict, Any, Optional, Tuple

class FaceRecognizer:
    """Handles face recognition using ArcFace."""
    
    def __init__(self, known_people_dir: str = "known_people", reid_interval: float = 20.0):
        """Initialize the face recognizer.
        
        Args:
            known_people_dir: Directory containing known people's face images
            reid_interval: Time interval in seconds between re-identification attempts
        """
        # Initialize ArcFace model
        self.face_model = insightface.app.FaceAnalysis(name='buffalo_l')
        self.face_model.prepare(ctx_id=0)  # Use GPU if available
        
        # Load known people database
        self.known_people_dir = known_people_dir
        self.known_people = {}  # {name: {'features': [feature_vectors], 'images': [image_paths]}}
        self.similarity_threshold = 0.55  # Cosine similarity threshold
        self.confidence_threshold = 0.4  # Minimum confidence to override existing identity
        self.load_known_people()
        
        # Re-identification tracking
        self.reid_interval = reid_interval
        self.last_reid_time = {}  # {track_id: last_reid_time}
        self.track_recognition_cache = {}  # {track_id: {'name': str, 'feature': np.ndarray, 'confidence': float}}
        
    def load_known_people(self) -> None:
        """Load known people's face features from the database."""
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
                    # Load and process image
                    frame = cv2.imread(img_path)
                    if frame is None:
                        continue
                        
                    # Detect faces
                    faces = self.face_model.get(frame)
                    if not faces:
                        continue
                    
                    # Use best face
                    best_face = max(faces, key=lambda x: x.det_score)
                    embedding = best_face.embedding
                    normalized_embedding = embedding / np.linalg.norm(embedding)
                    
                    features.append(normalized_embedding)
                    images.append(img_path)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

            if features:
                self.known_people[person_name] = {
                    'features': features,
                    'images': images
                }
    
    def extract_face_feature(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Extract face feature from a person's bounding box.
        
        Args:
            frame: Input image frame
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Normalized face feature vector or None if no face detected
        """
        try:
            # Extract face region with padding
            x1, y1, x2, y2 = map(int, bbox)
            height = y2 - y1
            width = x2 - x1
            pad_x = int(width * 0.2)
            pad_y = int(height * 0.2)
            
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(frame.shape[1], x2 + pad_x)
            y2 = min(frame.shape[0], y2 + pad_y)
            
            if (x2 - x1) < 30 or (y2 - y1) < 30:
                return None

            face_img = frame[y1:y2, x1:x2]
            faces = self.face_model.get(face_img)
            
            if not faces:
                return None
                
            # Get best face embedding
            best_face = max(faces, key=lambda x: x.det_score)
            embedding = best_face.embedding
            return embedding / np.linalg.norm(embedding)
            
        except Exception as e:
            print(f"Error extracting face feature: {e}")
            return None
    
    def update(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Recognize faces in detected people.
        
        Args:
            frame: Input image frame
            detections: List of person detections
            
        Returns:
            Updated detections with recognition info
        """
        current_time = time.time()
        recognized_detections = []
        
        for det in detections:
            track_id = det.get('track_id')
            bbox = det['bbox']
            
            # Check if we should skip re-identification for this track
            should_skip_reid = False
            if track_id is not None:
                last_reid = self.last_reid_time.get(track_id, 0)
                if current_time - last_reid < self.reid_interval:
                    should_skip_reid = True
                    print(f"SKIPPED re-identification for track {track_id}")
            
            if should_skip_reid and track_id in self.track_recognition_cache:
                # Use cached recognition results
                cached_data = self.track_recognition_cache[track_id]
                det['face_feature'] = cached_data['feature']
                det['recognized_name'] = cached_data['name']
                recognized_detections.append(det)
                continue
            
            # Perform face recognition
            face_feature = self.extract_face_feature(frame, bbox)
            
            if face_feature is not None:
                # Successfully extracted face feature
                best_match, confidence = self._find_best_match_with_confidence(face_feature)
                
                # Check if we should update the identity
                should_update_identity = True
                if track_id is not None and track_id in self.track_recognition_cache:
                    cached_data = self.track_recognition_cache[track_id]
                    cached_name = cached_data['name']
                    cached_confidence = cached_data.get('confidence', 0.0)
                    
                    # Only update if:
                    # 1. Current confidence is significantly higher, OR
                    # 2. Previous was "Unknown" and current is known, OR  
                    # 3. Current confidence is above threshold and previous was low confidence
                    if (cached_name != "Unknown" and 
                        confidence < cached_confidence + 0.1 and 
                        confidence < self.confidence_threshold):
                        # Keep previous identity if current recognition is not confident enough
                        det['face_feature'] = cached_data['feature']
                        det['recognized_name'] = cached_name
                        print(f"Low confidence ({confidence:.3f}) for track {track_id}, keeping previous: {cached_name}")
                        should_update_identity = False
                
                if should_update_identity:
                    det['face_feature'] = face_feature
                    det['recognized_name'] = best_match
                    
                # Update cache and timestamp
                if track_id is not None:
                    self.last_reid_time[track_id] = current_time
                    if should_update_identity:
                        self.track_recognition_cache[track_id] = {
                            'name': best_match,
                            'feature': face_feature,
                            'confidence': confidence
                        }
                    
            else:
                # Failed to extract face feature
                if track_id is not None and track_id in self.track_recognition_cache:
                    # Keep previous recognition if we had one
                    cached_data = self.track_recognition_cache[track_id]
                    det['face_feature'] = cached_data['feature']
                    det['recognized_name'] = cached_data['name']
                    print(f"Face not visible for track {track_id}, keeping previous recognition: {cached_data['name']}")
                    
                    # Update timestamp but keep the same recognition
                    self.last_reid_time[track_id] = current_time
                else:
                    # No previous recognition available
                    det['face_feature'] = None
                    det['recognized_name'] = "Unknown"
                    
                    if track_id is not None:
                        self.last_reid_time[track_id] = current_time
                        self.track_recognition_cache[track_id] = {
                            'name': "Unknown",
                            'feature': None,
                            'confidence': 0.0
                        }
                
            recognized_detections.append(det)
            
        return recognized_detections
    
    def cleanup_old_tracks(self, active_track_ids: List[int]) -> None:
        """Clean up recognition cache for tracks that no longer exist.
        
        Args:
            active_track_ids: List of currently active track IDs
        """
        # Remove cached data for tracks that no longer exist
        cached_track_ids = list(self.track_recognition_cache.keys())
        for track_id in cached_track_ids:
            if track_id not in active_track_ids:
                del self.track_recognition_cache[track_id]
                if track_id in self.last_reid_time:
                    del self.last_reid_time[track_id]
    
    def _find_best_match_with_confidence(self, face_feature: np.ndarray) -> Tuple[str, float]:
        """Find best matching person from known people database with confidence score.
        
        Args:
            face_feature: Face feature vector to match
            
        Returns:
            Tuple of (name of best matching person or "Unknown", confidence score)
        """
        best_name = "Unknown"
        best_similarity = -1
        
        for name, data in self.known_people.items():
            for known_feature in data['features']:
                similarity = np.dot(face_feature, known_feature)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_name = name
        
        final_name = best_name if best_similarity > self.similarity_threshold else "Unknown"
        return final_name, best_similarity
    
    def _find_best_match(self, face_feature: np.ndarray) -> str:
        """Find best matching person from known people database.
        
        Args:
            face_feature: Face feature vector to match
            
        Returns:
            Name of best matching person or "Unknown"
        """
        name, _ = self._find_best_match_with_confidence(face_feature)
        return name
    
    def visualize(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw recognition results on the frame.
        
        Args:
            frame: Input image frame
            detections: List of recognized detections
            
        Returns:
            Frame with visualization
        """
        vis_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            name = det.get('recognized_name', 'Unknown')
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw name
            cv2.putText(vis_frame, name, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)