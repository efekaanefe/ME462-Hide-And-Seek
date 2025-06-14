#!/usr/bin/env python3
from utils.person_detector import PersonDetector
from utils.face_recognizer import FaceRecognizer
from utils.homography_projector import HomographyProjector
from utils.orientation_detector import OrientationDetector
from utils.person_tracker import PersonTracker
from utils.MQTTPublisher import MQTTPublisher
from utils.UDPClient import UDPClient 
import json
import cv2
import time 


def run_tracking_with_udp(host: str, port: int = 8080, output_path: str = None, 
                         room_index: int = 0, cam_index: int = 0,
                         headless: bool = False, show_fps: bool = True,
                         use_simple_client: bool = False):
    """Run tracking with UDP stream input instead of video file
    
    Args:
        host: UDP server IP address
        port: UDP server port
        output_path: Path to save output video (optional)
        room_index: Room index to use
        cam_index: Camera index to use
        headless: If True, run without displaying video window
        show_fps: If True, display FPS and latency on video
        use_simple_client: If True, use SimpleUDPClient for JPEG frames
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
    
    # Initialize UDP client instead of TCP client
    
    udp_client = UDPClient(host, port)
    print("Using UDPClient for pickled frames")
        
    if not udp_client.connect():
        print("Failed to initialize UDP client")
        return
    
    # Video writer setup (if output path provided)
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (1600, 1200))  # Adjust fps as needed

    # Initialize visualization window if not headless
    if not headless:
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking", 1280, 720)
        
    # Initialize FPS and latency calculation
    fps_list = []
    latency_list = []
    frame_count = 0
    consecutive_timeouts = 0
    max_consecutive_timeouts = 10  # Reconnect after this many timeouts
    
    print("Starting live UDP tracking... Press 'q' to quit")
    
    while udp_client.is_connected():
        # Get frame from UDP stream instead of video file
        frame = udp_client.get_frame()
        if frame is None:
            consecutive_timeouts += 1
            print(f"No frame received, timeout #{consecutive_timeouts}")
            
            if consecutive_timeouts >= max_consecutive_timeouts:
                print("Too many consecutive timeouts, reconnecting...")
                udp_client.disconnect()
                time.sleep(1)
                if not udp_client.connect():
                    break
                consecutive_timeouts = 0
            continue
        
        # Reset timeout counter on successful frame
        consecutive_timeouts = 0
            
        # Start timing for FPS calculation
        frame_start_time = time.time()
        
        # Process frame (same as original)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.resize(frame, (1600, 1200))

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

        # Print results (reduce frequency for UDP to avoid spam)
        if frame_count % 30 == 0:  # Print every 30 frames
            print("-"*30)
            for step, t in times.items():
                print(f"{step}: {t:.2f} ms ({(t / total_time * 100):.2f}%)")
            print(f"Total pipeline time: {total_time:.2f} ms")
            print("-"*30)

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
    udp_client.disconnect()
    if out:
        out.release()
    publisher.disconnect()
    if not headless:
        cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames")
    if fps_list:
        print(f"Average FPS: {fps:.1f}")
        print(f"Average Latency: {latency_ms:.1f}ms")

def main():
    # Example usage - replace with your Pi's IP
    run_tracking_with_udp(
        host="192.168.68.59",  # Your Raspberry Pi IP
        port=8080,
        output_path="output_tracking_live_udp.mp4",  # Optional
        room_index=0,
        cam_index=0,
        headless=False,
        show_fps=True,
        use_simple_client=False  # Set to True if using JPEG compression on server
    )

if __name__ == "__main__":
    main()
