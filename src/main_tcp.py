from utils import PersonDetector
from utils import FaceRecognizer
from utils import HomographyProjector
from utils import OrientationDetector
from utils import PersonTracker
from utils import MQTTPublisher
from utils import TCPClient
from utils import FrameBuffer, frame_reader_thread
import json
import cv2
import time
import argparse 
import threading
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
                         headless: bool = False, show_fps: bool = True,
                         use_queue: bool = True, buffer_size: int = 30):
    """Run tracking with TCP stream input using threaded frame reading
    
    Args:
        host: TCP server IP address
        port: TCP server port
        output_path: Path to save output video (optional)
        room_index: Room index to use
        cam_index: Camera index to use
        headless: If True, run without displaying video window
        show_fps: If True, display FPS and latency on video
        use_queue: If True, use FIFO queue; if False, use LIFO stack
        buffer_size: Maximum number of frames to buffer
    """

    detector = PersonDetector()
    recognizer = FaceRecognizer()
    projector = HomographyProjector()
    orientation = OrientationDetector()
    tracker = PersonTracker()
    publisher = MQTTPublisher(broker_address="test.mosquitto.org", room_index=room_index, camera_index=cam_index)
    publisher.connect()
    
    # Wait for MQTT connection
    time.sleep(2)
    
    projector.select_room(room_index)
    projector.select_camera(cam_index)
    
    tcp_client = TCPClient(host, port)
    if not tcp_client.connect():
        print("Failed to connect to TCP server")
        return
    
    # Initialize frame buffer
    frame_buffer = FrameBuffer(maxsize=buffer_size, use_queue=use_queue)
    
    # Start frame reader thread
    stop_event = threading.Event()
    reader_thread = threading.Thread(
        target=frame_reader_thread, 
        args=(tcp_client, frame_buffer, stop_event),
        daemon=True
    )
    reader_thread.start()
    
    # Video writer setup (if output path provided)
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (1920, 1080))

    # Initialize visualization window if not headless
    if not headless:
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking", 1920, 1080)
        
    # Initialize FPS and latency calculation
    fps_list = []
    frame_count = 0
    
    buffer_type = "Queue (FIFO)" if use_queue else "Stack (LIFO)"
    print(f"Starting live tracking with {buffer_type} buffer... Press 'q' to quit")
    
    try:
        while True:
            # Get frame from buffer
            frame = frame_buffer.get()
            if frame is None:
                # No frame available, wait a bit
                time.sleep(0.001)
                continue
                
            # Start timing for FPS calculation
            frame_start_time = time.time()
          
            frame = cv2.resize(frame, (1920, 1080))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Your original processing pipeline
            start = time.perf_counter()

            detections = detector.update(frame)

            tracks = tracker.update(detections, time.time())

            detections = recognizer.update(frame, detections)

            #detections = orientation.update(frame, detections)

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
                buffer_info = f"Buffer: {frame_buffer.size()}/{buffer_size} ({buffer_type})"
                cv2.putText(vis_frame, f"FPS: {fps:.1f}", (1300, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(vis_frame, buffer_info, (1300, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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
                    
            frame_count += 1
                    
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Cleanup
        print("Shutting down...")
        stop_event.set()
        reader_thread.join(timeout=2)
        tcp_client.disconnect()
        if out:
            out.release()
        publisher.disconnect()
        if not headless:
            cv2.destroyAllWindows()
        
        print(f"Processed {frame_count} frames")
        if fps_list:
            print(f"Average FPS: {sum(fps_list)/len(fps_list):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Run tracking for a specific room and camera.")
    parser.add_argument("--room", type=int, default=0, help="Index of the room (default: 0)")
    parser.add_argument("--cam", type=int, default=0, help="Index of the camera (default: 0)")
    parser.add_argument("--use-stack", action="store_true", help="Use LIFO stack instead of FIFO queue")
    parser.add_argument("--buffer-size", type=int, default=30, help="Frame buffer size (default: 30)")
    args = parser.parse_args()

    room_index = args.room
    camera_index = args.cam
    room_str = f"room{room_index}"
    camera_str = f"cam{camera_index}"
    ip = get_camera_ip(room_str, camera_str)

    run_tracking_with_tcp(
        host=ip,
        port=8080,
        output_path=None,
        room_index=room_index,
        cam_index=camera_index,
        headless=False,
        show_fps=True,
        use_queue=not args.use_stack,  # If --use-stack is set, use_queue becomes False
        buffer_size=args.buffer_size,
    )

if __name__ == "__main__":
    main()
