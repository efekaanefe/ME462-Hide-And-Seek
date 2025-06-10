import cv2
import numpy as np
import insightface
import os
import time
from typing import List, Dict, Any, Optional, Tuple

class FaceRecognizer:
    """Handles face recognition using ArcFace."""
    
    def __init__(self, known_people_dir: str = "known_people", reid_interval: float = 5.0):
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
        self.load_known_people()
        
        # Re-identification tracking
        self.reid_interval = reid_interval
        self.last_reid_time = {}  # {track_id: last_reid_time}
        
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
            if track_id is not None:
                last_reid = self.last_reid_time.get(track_id, 0)
                if current_time - last_reid < self.reid_interval:
                    # Keep existing recognition if available
                    if 'recognized_name' in det:
                        recognized_detections.append(det)
                        continue
            
            # Perform face recognition
            face_feature = self.extract_face_feature(frame, bbox)
            
            if face_feature is not None:
                # Find best match in known people
                best_match = self._find_best_match(face_feature)
                det['face_feature'] = face_feature
                det['recognized_name'] = best_match
                
                # Update last re-identification time
                if track_id is not None:
                    self.last_reid_time[track_id] = current_time
            else:
                det['face_feature'] = None
                det['recognized_name'] = "Unknown"
                
            recognized_detections.append(det)
            
        return recognized_detections
    
    def _find_best_match(self, face_feature: np.ndarray) -> str:
        """Find best matching person from known people database.
        
        Args:
            face_feature: Face feature vector to match
            
        Returns:
            Name of best matching person or "Unknown"
        """
        best_name = "Unknown"
        best_similarity = -1
        
        for name, data in self.known_people.items():
            for known_feature in data['features']:
                similarity = np.dot(face_feature, known_feature)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_name = name
        
        return best_name if best_similarity > self.similarity_threshold else "Unknown"
    
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
        
        return vis_frame 