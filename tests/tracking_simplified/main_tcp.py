#!/usr/bin/env python3
from utils import PersonDetector
from utils import FaceRecognizer
from utils import HomographyProjector
from utils import OrientationDetector
from utils import PersonTracker
from utils import MQTTPublisher
from utils import TCPClient

import json
import cv2
import time
import argparse 

import configparser

def get_camera_ip(room: str, camera: str, config_path="ip_config.ini") -> str:
    config = configparser.ConfigParser()
    config.read(config_path)

    try:
        return config[room][camera]
    except KeyError:
        raise ValueError(f"No IP found for {room}.{camera}")


def run_tracking_with_tcp(host: str, port: int = 8080, output_path: str = None, 
                         room_index: int = 0, cam_index: int = 0,
                         headless: bool = False, show_fps: bool = True):
    """Run tracking with TCP stream input instead of video file
    
    Args:
        host: TCP server IP address
        port: TCP server port
        output_path: Path to save output video (optional)
        room_index: Room index to use
        cam_index: Camera index to use
        headless: If True, run without displaying video window
        show_fps: If True, display FPS and latency on video
    """

    detector = PersonDetector()
    recognizer = FaceRecognizer()
    projector = HomographyProjector()
    orientation = OrientationDetector()
    tracker = PersonTracker()
    publisher = MQTTPublisher(broker_address="mqtt.eclipseprojects.io", room_index=room_index, camera_index=cam_index)
    publisher.connect()
    
    # Wait for MQTT connection
    time.sleep(2)
    
    projector.select_room(room_index)
    projector.select_camera(cam_index)
    
    tcp_client = TCPClient(host, port)
    if not tcp_client.connect():
        print("Failed to connect to TCP server")
        return
    
    # Video writer setup (if output path provided)
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (1920, 1080))  # Adjust fps as needed

    # Initialize visualization window if not headless
    if not headless:
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking", 1920, 1080)
        
    # Initialize FPS and latency calculation
    fps_list = []
    latency_list = []
    frame_count = 0
    
    print("Starting live tracking... Press 'q' to quit")
    
    while tcp_client.is_connected():
        # Get frame from TCP stream instead of video file
        frame = tcp_client.get_frame()
        if frame is None:
            print("No frame received, reconnecting...")
            tcp_client.disconnect()
            time.sleep(1)
            if not tcp_client.connect():
                break
            continue
            
        # Start timing for FPS calculation
        frame_start_time = time.time()
      
        frame = cv2.resize(frame, (1920, 1080))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Your original processing pipeline
        start = time.perf_counter()

        detections = detector.update(frame)

        tracks = tracker.update(detections, time.time())

        print("-"*50)
        print("Before recognizer:", detections[0].keys() if detections else "No detections")
        detections = recognizer.update(frame, detections)
        print("After recognizer:", detections[0].keys() if detections else "No detections")

        detections = orientation.update(frame, detections)

        detections = projector.update(detections)

        # Update tracker again
        tracks = tracker.update(detections, time.time())

        # Clean up old recognition cache entries
        active_track_ids = list(tracks.keys())
        recognizer.cleanup_old_tracks(active_track_ids)


        frame_time = time.time() - frame_start_time
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        fps_list.append(current_fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
        fps = sum(fps_list) / len(fps_list)
        
        vis_frame = frame.copy()
        if show_fps:
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (1300, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        vis_frame = tracker.visualize(vis_frame, tracks)
        vis_frame = projector.visualize(vis_frame, list(tracks.values()))
        
        # MQTT publishing 
        if publisher.is_connected:
            for track_id, track in tracks.items():
                if 'map_position' in track:
                    track_data = {
                        "track_id": track_id,
                        "name": track.get('name', 'Unknown'),
                        "x": float(track['map_position'][0]),
                        "y": float(track['map_position'][1]),
                        "orientation": float(track['orientation']) if 'orientation' in track else None,
                        "timestamp": time.time()
                    }

                    publisher.publish(
                        f"tracking/{room_index}/{cam_index}/{track_id}", 
                        json.dumps(track_data), 
                        qos=1
                    )

                    print(f"{track['map_position'][0]}, {track['map_position'][1]}")
        
        # Write frame to output video (if enabled)
        if out:
            out.write(vis_frame)
        
        # Display frame if not headless
        if not headless:
            cv2.imshow("Tracking", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    # Cleanup (same as original)
    tcp_client.disconnect()
    if out:
        out.release()
    publisher.disconnect()
    if not headless:
        cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames")
    print(f"Average FPS: {fps:.1f}")


def main():
<<<<<<< HEAD
    room_index = 0
    camera_index = 0
    ip = get_camera_ip(room_index, camera_index)
=======
    parser = argparse.ArgumentParser(description="Run tracking for a specific room and camera.")
    parser.add_argument("--room", type=int, default=0, help="Index of the room (default: 0)")
    parser.add_argument("--cam", type=int, default=2, help="Index of the camera (default: 2)")
    args = parser.parse_args()

    room_index = args.room
    camera_index = args.cam
    room_str = f"room{room_index}"
    camera_str = f"cam{camera_index}"
    ip = get_camera_ip(room_str, camera_str)
>>>>>>> 2990e1d2 (manager and other codes are tested, homography calibration is done)

    run_tracking_with_tcp(
        host=ip,
        port=8080,
        output_path=None,
        room_index=room_index,
        cam_index=camera_index,
        headless=False,
        show_fps=True,
    )

if __name__ == "__main__":
    main()