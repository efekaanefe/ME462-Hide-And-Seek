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

from yolo_labels import get_class_ids

ensure_packages_installed(['ultralytics', 'cv2'])


"""

    Usage:
        python3 motpy_tracker_yolo.py \
            --video_path=video.mp4 \
            --detect_labels="car,truck,person" \
            --tracker_min_iou=0.2 \
            --yolo_model=yolov8n.pt \
            --device=cpu

    Available YOLO models:
        - yolov8n.pt (nano - fastest, least accurate)
        - yolov8s.pt (small)
        - yolov8m.pt (medium)  
        - yolov8l.pt (large)
        - yolov8x.pt (extra large - slowest, most accurate)

"""


logger = setup_logger(__name__, level="DEBUG", is_main=True) # level="DEBUG"


class YoloObjectDetector(BaseObjectDetector):
    """ A wrapper for YOLO object detector using ultralytics """

    def __init__(self,
                 class_ids: Sequence[int],
                 confidence_threshold: float = 0.5,
                 yolo_model: str = 'yolov8n.pt',
                 device: str = 'cpu'):

        self.confidence_threshold = confidence_threshold
        self.device = device
        self.class_ids = class_ids
        assert len(self.class_ids) > 0, f'select more than one class_ids'

        # Initialize YOLO model
        self.model = YOLO(yolo_model)
        
        # Set device
        if device != 'cpu':
            self.model.to(device)
            
    def process_image(self, image: NpImage) -> Sequence[Detection]:
        t0 = time.time()
        
        # Run inference
        results = self.model(image, verbose=False, conf=self.confidence_threshold)
        
        detections = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                # Get predictions
                xyxy = boxes.xyxy.cpu().numpy()  # bounding boxes in xyxy format
                conf = boxes.conf.cpu().numpy()  # confidence scores
                cls = boxes.cls.cpu().numpy().astype(int)  # class ids
                
                # Filter by selected class IDs
                mask = np.isin(cls, self.class_ids)
                filtered_count = np.sum(mask)
                
                for i in range(len(xyxy)):
                    if mask[i]:
                        box = xyxy[i]  # [x1, y1, x2, y2]
                        score = conf[i]
                        class_id = cls[i]
                        
                        detections.append(Detection(
                            box=box,
                            score=score,
                            class_id=class_id
                        ))
        
        elapsed = (time.time() - t0) * 1000.
        logger.debug(f'inference time: {elapsed:.3f} ms, detections: {len(detections)}')
        
        return detections


def read_video_file(video_path: str):
    video_path = os.path.expanduser(video_path)
    cap = cv2.VideoCapture(video_path)
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    return cap, video_fps


def run(video_path: str, detect_labels,
        video_downscale: float = 1.,
        yolo_model: str = 'yolov8n.pt',
        confidence_threshold: float = 0.4,
        tracker_min_iou: float = 0.25,
        show_detections: bool = False,
        track_text_verbose: int = 0,
        device: str = 'cpu',
        viz_wait_ms: int = 1):
    
    # setup detector, video reader and object tracker
    detector = YoloObjectDetector(
        class_ids=get_class_ids(detect_labels), 
        confidence_threshold=confidence_threshold, 
        yolo_model=yolo_model, 
        device=device
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
