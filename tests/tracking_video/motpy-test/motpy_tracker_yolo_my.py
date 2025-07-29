import os
import time
from typing import Sequence

import cv2
import fire
import numpy as np
from motpy import Detection, ModelPreset, MultiObjectTracker, NpImage
from motpy.core import setup_logger
from motpy.detector import BaseObjectDetector
from motpy.testing_viz import draw_detection, draw_track
from motpy.utils import ensure_packages_installed
from ultralytics import YOLO

ensure_packages_installed(['ultralytics', 'cv2'])


"""

    Usage:
        python3 motpy_tracker_yolo_my.py \
            --video_path=video3.mp4 \
            --confidence_threshold=0.5 \
            --tracker_min_iou=0.2 \
            --yolo_model=yolov8n.pt \
            --device=cpu

"""


logger = setup_logger(__name__, level="DEBUG", is_main=True)


class PersonDetector:
    """Handles person detection using YOLOv8."""
    
    def __init__(self, model_path: str = 'yolov8n.pt'):
        """Initialize the detector with YOLOv8 model.
        
        Args:
            model_path: Path to YOLOv8 model weights
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = 0.65
        
    def update(self, frame: np.ndarray):
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
            if boxes is not None:
                for box in boxes:
                    if box.conf > self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(box.conf),
                            'class': int(box.cls)
                        })
        
        return detections
    
    def visualize(self, frame: np.ndarray, detections):
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


class PersonDetectorWrapper(BaseObjectDetector):
    """Wrapper to make PersonDetector compatible with MOTPY"""
    
    def __init__(self, yolo_model: str = 'yolov8n.pt', confidence_threshold: float = 0.65):
        self.detector = PersonDetector(model_path=yolo_model)
        self.detector.confidence_threshold = confidence_threshold
        
    def process_image(self, image: NpImage) -> Sequence[Detection]:
        t1 = time.time()
        
        # Use PersonDetector to get detections
        detections_dict = self.detector.update(image)
        
        # Convert to MOTPY Detection format
        detections = []
        for det in detections_dict:
            bbox = det['bbox']  # [x1, y1, x2, y2]
            score = det['confidence']
            class_id = det['class']
            
            detections.append(Detection(
                box=bbox,
                score=score,
                class_id=class_id
            ))
        
        elapsed = (time.time() - t1) * 1000.
        logger.debug(f'inference time: {elapsed:.4f} ms, detections: {len(detections)}')
        
        return detections


def read_video_file(video_path: str):
    video_path = os.path.expanduser(video_path)
    cap = cv2.VideoCapture(video_path)
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    return cap, video_fps


def run(video_path: str,
        video_downscale: float = 1.,
        yolo_model: str = 'yolov8n.pt',
        confidence_threshold: float = 0.65,
        tracker_min_iou: float = 0.25,
        show_detections: bool = False,
        track_text_verbose: int = 0,
        device: str = 'cpu',
        viz_wait_ms: int = 1):
    
    # setup detector, video reader and object tracker
    detector = PersonDetectorWrapper(
        yolo_model=yolo_model,
        confidence_threshold=confidence_threshold
    )
    
    cap, cap_fps = read_video_file(video_path)
    
    tracker = MultiObjectTracker(
        dt=1 / cap_fps,
        tracker_kwargs={'max_staleness': 5},
        model_spec={'order_pos': 1, 'dim_pos': 2,
                    'order_size': 0, 'dim_size': 2,
                    'q_var_pos': 5000., 'r_var_pos': 0.1},
        matching_fn_kwargs={'min_iou': tracker_min_iou,
                            'multi_match_min_iou': 0.93})

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, fx=video_downscale, fy=video_downscale, dsize=None, interpolation=cv2.INTER_AREA)

        # detect objects in the frame
        detections = detector.process_image(frame)

        # track detected objects
        _ = tracker.step(detections=detections)
        active_tracks = tracker.active_tracks(min_steps_alive=3)

        # visualize and show detections and tracks
        if show_detections:
            for det in detections:
                draw_detection(frame, det)

        for track in active_tracks:
            draw_track(frame, track, thickness=2, text_at_bottom=True, text_verbose=track_text_verbose)

        cv2.imshow('frame', frame)
        c = cv2.waitKey(viz_wait_ms)
        if c == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    fire.Fire(run)
