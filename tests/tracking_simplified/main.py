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
        
        
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) # Rotate 
        frame = cv2.resize(frame, (1600, 1200)) # Resize frame to 1600x1200

        # TODO: why fps is around 2?

        # Process frame through pipeline
        detections = detector.update(frame)
        detections = recognizer.update(frame, detections)
        detections = orientation.update(frame, detections)
        detections = projector.update(detections)
        tracks = tracker.update(detections, time.time())
        
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
        
        print(tracks)
        
        # Publish tracking data to MQTT
        # if publisher.is_connected:
        for track_id, track in tracks.items():
            print(track)
            if 'map_position' in track:
                position_data = {
                    "track_id": track_id,
                    "name": track.get('name', 'Unknown'),
                    "x": float(track['map_position'][0]),
                    "y": float(track['map_position'][1]),
                    "orientation": float(track['orientation']) if 'orientation' in track else None,
                    "timestamp": time.time()
                }
                # publisher.publish(
                #     f"game/player/position/{track_id}", 
                #     json.dumps(position_data), 
                #     qos=1
                # )
                print(f"{track['map_position'][0]}, {track['map_position'][1]}")


        
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