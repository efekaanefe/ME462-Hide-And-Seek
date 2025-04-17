
import cv2
import time
import numpy as np
import torch

from mmdet.apis import init_detector, inference_detector
from mmpose.apis.inference import init_model, inference_top_down_pose_model
from mmpose.apis import vis_pose_result

def main():
    # Model config and checkpoint paths (change these to match your files)
    det_config = 'mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    det_checkpoint = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    
    pose_config = 'mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
    pose_checkpoint = 'checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("Loading detection model...")
    det_model = init_detector(det_config, det_checkpoint, device=device)

    print("Loading pose estimation model...")
    pose_model = init_model(pose_config, pose_checkpoint, device=device)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    bbox_thr = 0.3  # detection threshold

    print("Running pose estimation... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # Detect humans in frame
        mmdet_results = inference_detector(det_model, frame)

        # Only keep person class (COCO class 0)
        person_results = []
        if isinstance(mmdet_results, tuple):
            mmdet_results = mmdet_results[0]
        bboxes = mmdet_results[0]  # class 0: person
        for bbox in bboxes:
            if bbox[4] >= bbox_thr:
                person_results.append({'bbox': bbox})

        # Pose estimation
        pose_results = inference_top_down_pose_model(
            pose_model,
            frame,
            person_results,
            bbox_thr=bbox_thr,
            format='xyxy',
            dataset=pose_model.cfg.data.test.type
        )

        # Visualize
        vis_frame = vis_pose_result(
            pose_model,
            frame,
            pose_results,
            kpt_score_thr=0.3,
            radius=4,
            thickness=2
        )

        cv2.imshow('Pose Estimation', vis_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
