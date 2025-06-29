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

    # Initialize components (same as original)
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
    
    # Initialize TCP client instead of VideoCapture
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
        
        # Process frame (same as original)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #frame = cv2.resize(frame, (1600, 1200))
        frame = cv2.resize(frame, (1920, 1080))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Your original processing pipeline
        start = time.perf_counter()

        # Step 1: Detector
        t0 = time.perf_counter()
        detections = detector.update(frame)
        t1 = time.perf_counter()

        # Step 2: Tracker
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

        # Step 6: Update tracker again
        tracks = tracker.update(detections, time.time())

        # Step 7: Clean up old recognition cache entries
        active_track_ids = list(tracks.keys())
        recognizer.cleanup_old_tracks(active_track_ids)

        # Timing calculations (same as original)
        times = {
            "Detector": (t1 - t0) * 1000,
            "Tracker": (t2 - t1) * 1000,
            "Recognizer": (t3 - t2) * 1000,
            "Orientation": (t4 - t3) * 1000,
            "Projector": (t5 - t4) * 1000
        }

        total_time = sum(times.values())
        frame_count += 1

        # # Print results
        # print("-"*30)
        # for step, t in times.items():
        #     print(f"{step}: {t:.2f} ms ({(t / total_time * 100):.2f}%)")
        # print(f"Total pipeline time: {total_time:.2f} ms")
        # print("-"*30)

        # Calculate FPS and latency
        frame_time = time.time() - frame_start_time
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        current_latency_ms = frame_time * 1000  # Convert to milliseconds
        
        # Keep rolling averages
        fps_list.append(current_fps)
        latency_list.append(current_latency_ms)
        if len(fps_list) > 30:
            fps_list.pop(0)
        if len(latency_list) > 30:
            latency_list.pop(0)
            
        fps = sum(fps_list) / len(fps_list)
        latency_ms = sum(latency_list) / len(latency_list)
        
        # Draw visualizations (same as original)
        vis_frame = frame.copy()
        if show_fps:
            # Display FPS
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (1300, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Display latency below FPS
            cv2.putText(vis_frame, f"Latency: {latency_ms:.1f}ms", (1300, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                       
        # Draw tracking results
        vis_frame = tracker.visualize(vis_frame, tracks)
        
        # Draw map visualization using projector
        vis_frame = projector.visualize(vis_frame, list(tracks.values()))
        
        # MQTT publishing (same as original, uncomment if needed)
        # if publisher.is_connected:
        #     for track_id, track in tracks.items():
        #         if 'map_position' in track:
        #             position_data = {
        #                 "track_id": track_id,
        #                 "name": track.get('name', 'Unknown'),
        #                 "x": float(track['map_position'][0]),
        #                 "y": float(track['map_position'][1]),
        #                 "orientation": float(track['orientation']) if 'orientation' in track else None,
        #                 "timestamp": time.time()
        #             }
        #             print(f"{track['map_position'][0]}, {track['map_position'][1]}")
        
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
    print(f"Average Latency: {latency_ms:.1f}ms")

def main():
    room = "room0"
    camera = "cam1"
    ip = get_camera_ip(room, camera)

    run_tracking_with_tcp(
        host=ip,  # Your Raspberry Pi IP
        port=8080,
        output_path="output_tracking_live.mp4",  # Optional
        room_index=0,
        cam_index=0,
        headless=False,
        show_fps=True
    )

if __name__ == "__main__":
    main()
