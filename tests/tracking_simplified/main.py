import cv2
import time
import json
from person_detector import PersonDetector
from face_recognizer import FaceRecognizer
from homography_projector import HomographyProjector
from orientation_detector import OrientationDetector
from person_tracker import PersonTracker
from MQTTPublisher import MQTTPublisher
import numpy as np

def run_tracking(video_path: str, output_path: str, room_index: int = 0, cam_index: int = 0,
                headless: bool = False, show_fps: bool = True):
    """Run the complete tracking system.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save output video
        room_index: Room index to use
        cam_index: Camera index to use
        headless: If True, run without displaying video window
        show_fps: If True, display FPS on video
    """
    # Initialize components
    detector = PersonDetector()
    recognizer = FaceRecognizer()
    projector = HomographyProjector()
    orientation = OrientationDetector()
    tracker = PersonTracker()
    publisher = MQTTPublisher(broker_address="mqtt.eclipseprojects.io")
    publisher.connect()
    
    # Wait for MQTT connection
    time.sleep(2)
    
    # Select room and camera
    projector.select_room(room_index)
    projector.select_camera(cam_index)
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
        
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    out = cv2.VideoWriter(output_path, fourcc, fps, (1600, 1200))

    # Initialize visualization window if not headless
    if not headless:
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking", 1280, 720)
        
    # Initialize FPS calculation
    fps_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Start timing for FPS calculation
        frame_start_time = time.time()
        
        
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.resize(frame, (1600, 1200)) 



        # # Process frame through pipeline
        # detections = detector.update(frame)
        # detections = recognizer.update(frame, detections)
        # detections = orientation.update(frame, detections)
        # detections = projector.update(detections)
        # tracks = tracker.update(detections, time.time())


        # Detector: 46.85 ms (7.22%)
        # Recognizer: 501.35 ms (77.30%)
        # Orientation: 100.29 ms (15.46%)
        # Projector: 0.04 ms (0.01%)
        # Tracker: 0.06 ms (0.01%)

        # Detector: 48.17 ms (31.98%)
        # Tracker: 0.09 ms (0.06%)
        # Recognizer: 0.45 ms (0.30%)
        # Orientation: 101.84 ms (67.63%)
        # Projector: 0.04 ms (0.03%)
        # Total pipeline time: 150.60 ms


        start = time.perf_counter()

        # Step 1: Detector
        t0 = time.perf_counter()
        detections = detector.update(frame)
        t1 = time.perf_counter()

        # Step 2: Tracker (moved before recognizer to assign track_ids)
        tracks = tracker.update(detections, time.time())
        t2 = time.perf_counter()

        # Step 3: Recognizer 
        print("-"*50)
        print("Before recognizer:", detections[0].keys() if detections else "No detections")
        detections = recognizer.update(frame, detections)
        print("After recognizer:", detections[0].keys() if detections else "No detections")
        t3 = time.perf_counter()

        # Step 4: Orientation
        detections = orientation.update(frame, detections)
        t4 = time.perf_counter()

        # Step 5: Projector
        detections = projector.update(detections)
        t5 = time.perf_counter()

        # Step 6: Update tracker again with all the enriched detection data
        tracks = tracker.update(detections, time.time())

        # Step 7: Clean up old recognition cache entries
        active_track_ids = list(tracks.keys())
        recognizer.cleanup_old_tracks(active_track_ids)

        # Timing calculations (same as before)
        times = {
            "Detector": (t1 - t0) * 1000,
            "Tracker": (t2 - t1) * 1000,
            "Recognizer": (t3 - t2) * 1000,
            "Orientation": (t4 - t3) * 1000,
            "Projector": (t5 - t4) * 1000
        }

        total_time = sum(times.values())

        # Print performance stats every 30 frames (optional)
        frame_count = getattr(run_tracking, 'frame_count', 0) + 1
        run_tracking.frame_count = frame_count

        # Print results
        print("-"*30)
        for step, t in times.items():
            print(f"{step}: {t:.2f} ms ({(t / total_time * 100):.2f}%)")

        print(f"Total pipeline time: {total_time:.2f} ms")
        print("-"*30)




        # Calculate FPS
        frame_time = time.time() - frame_start_time
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        fps_list.append(current_fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
        fps = sum(fps_list) / len(fps_list)
        
        # Draw visualizations
        vis_frame = frame.copy()
        if show_fps:
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (1400, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                       
        # Draw tracking results
        vis_frame = tracker.visualize(vis_frame, tracks)
        
        # Draw map visualization using projector
        vis_frame = projector.visualize(vis_frame, list(tracks.values()))
        
        # print(tracks)
        
        # # Publish tracking data to MQTT
        # if publisher.is_connected:
        #     for track_id, track in tracks.items():
        #         print(track)
        #         if 'map_position' in track:
        #             position_data = {
        #                 "track_id": track_id,
        #                 "name": track.get('name', 'Unknown'),
        #                 "x": float(track['map_position'][0]),
        #                 "y": float(track['map_position'][1]),
        #                 "orientation": float(track['orientation']) if 'orientation' in track else None,
        #                 "timestamp": time.time()
        #             }
        #             # publisher.publish(
        #             #     f"game/player/position/{track_id}", 
        #             #     json.dumps(position_data), 
        #             #     qos=1
        #             # )
        #             print(f"{track['map_position'][0]}, {track['map_position'][1]}")


        
        # Write frame to output video
        out.write(vis_frame)
        
        # Display frame if not headless
        if not headless:
            cv2.imshow("Tracking", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    # Cleanup
    cap.release()
    out.release()
    publisher.disconnect()
    if not headless:
        cv2.destroyAllWindows()

def main():
    # Example usage
    run_tracking(
        video_path="test-home2.mp4",
        output_path="output_tracking.mp4",
        room_index=0,
        cam_index=0,
        headless=False,
        show_fps=True
    )

if __name__ == "__main__":
    main() 