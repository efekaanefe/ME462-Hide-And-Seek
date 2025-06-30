import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple

class PersonDetector:
    """Handles person detection using YOLOv8."""
    
    def __init__(self, model_path: str = 'models/yolov8n.pt'):
        """Initialize the detector with YOLOv8 model.
        
        Args:
            model_path: Path to YOLOv8 model weights
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = 0.3
        
    def update(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect people in the frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of dictionaries containing detection info:
            {
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'class': int
            }
        """
        # Run YOLO detection
        results = self.model(frame, classes=[0], verbose=False)  # class 0 is person
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.conf > self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(box.conf),
                        'class': int(box.cls)
                    })
        
        return detections
    
    def visualize(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection boxes on the frame.
        
        Args:
            frame: Input image frame
            detections: List of detection dictionaries
            
        Returns:
            Frame with visualization
        """
        vis_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            label = f"Person: {conf:.2f}"
            cv2.putText(vis_frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_frame 