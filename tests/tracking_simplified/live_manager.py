#!/usr/bin/env python3
import cv2
import socket
import pickle
import struct
import numpy as np
import time
import json
from person_detector import PersonDetector
from face_recognizer import FaceRecognizer
from homography_projector import HomographyProjector
from orientation_detector import OrientationDetector
from person_tracker import PersonTracker
from MQTTPublisher import MQTTPublisher

class LiveStreamTracker:
    def __init__(self, host, port=8080, room_index=0, cam_index=0, headless=False, show_fps=True):
        """Initialize the live stream tracker.
        
        Args:
            host: TCP server host (e.g., Raspberry Pi IP)
            port: TCP server port
            room_index: Room index to use
            cam_index: Camera index to use
            headless: If True, run without displaying video window
            show_fps: If True, display FPS on video
        """
        self.host = host
        self.port = port
        self.socket = None
        self.headless = headless
        self.show_fps = show_fps
        
        # Initialize tracking components
        self.detector = PersonDetector()
        self.recognizer = FaceRecognizer()
        self.projector = HomographyProjector()
        self.orientation = OrientationDetector()
        self.tracker = PersonTracker()
        
        # Initialize MQTT publisher
        self.publisher = MQTTPublisher(broker_address="mqtt.eclipseprojects.io")
        self.publisher.connect()
        time.sleep(2)  # Wait for MQTT connection
        
        # Select room and camera
        self.projector.select_room(room_index)
        self.projector.select_camera(cam_index)
        
        # Initialize FPS calculation
        self.fps_list = []
        self.frame_count = 0
        
        # Initialize visualization window if not headless
        if not self.headless:
            cv2.namedWindow("Live Tracking", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Live Tracking", 1280, 720)
    
    def recv_all(self, sock, n):
        """Receive exactly n bytes"""
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data
    
    def process_frame(self, frame):
        """Process a single frame through the tracking pipeline"""
        # Start timing for FPS calculation
        frame_start_time = time.time()
        
        # Rotate and resize frame (adjust as needed for your camera setup)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.resize(frame, (1600, 1200))
        
        # Performance timing
        start = time.perf_counter()
        
        # Step 1: Detector
        t0 = time.perf_counter()
        detections = self.detector.update(frame)
        t1 = time.perf_counter()
        
        # Step 2: Tracker (moved before recognizer to assign track_ids)
        tracks = self.tracker.update(detections, time.time())
        t2 = time.perf_counter()
        
        # Step 3: Recognizer
        print("-" * 50)
        print("Before recognizer:", detections[0].keys() if detections else "No detections")
        detections = self.recognizer.update(frame, detections)
        print("After recognizer:", detections[0].keys() if detections else "No detections")
        t3 = time.perf_counter()
        
        # Step 4: Orientation
        detections = self.orientation.update(frame, detections)
        t4 = time.perf_counter()
        
        # Step 5: Projector
        detections = self.projector.update(detections)
        t5 = time.perf_counter()
        
        # Step 6: Update tracker again with all the enriched detection data
        tracks = self.tracker.update(detections, time.time())
        
        # Step 7: Clean up old recognition cache entries
        active_track_ids = list(tracks.keys())
        self.recognizer.cleanup_old_tracks(active_track_ids)
        
        # Timing calculations
        times = {
            "Detector": (t1 - t0) * 1000,
            "Tracker": (t2 - t1) * 1000,
            "Recognizer": (t3 - t2) * 1000,
            "Orientation": (t4 - t3) * 1000,
            "Projector": (t5 - t4) * 1000
        }
        
        total_time = sum(times.values())
        
        # Print performance stats
        self.frame_count += 1
        print("-" * 30)
        for step, t in times.items():
            print(f"{step}: {t:.2f} ms ({(t / total_time * 100):.2f}%)")
        print(f"Total pipeline time: {total_time:.2f} ms")
        print("-" * 30)
        
        # Calculate FPS
        frame_time = time.time() - frame_start_time
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        self.fps_list.append(current_fps)
        if len(self.fps_list) > 30:
            self.fps_list.pop(0)
        fps = sum(self.fps_list) / len(self.fps_list)
        
        # Create visualization frame
        vis_frame = frame.copy()
        if self.show_fps:
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (1400, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw tracking results
        vis_frame = self.tracker.visualize(vis_frame, tracks)
        
        # Draw map visualization using projector
        vis_frame = self.projector.visualize(vis_frame, list(tracks.values()))
        
        # Publish tracking data to MQTT (uncomment if needed)
        if self.publisher.is_connected:
            for track_id, track in tracks.items():
                if 'map_position' in track:
                    position_data = {
                        "track_id": track_id,
                        "name": track.get('name', 'Unknown'),
                        "x": float(track['map_position'][0]),
                        "y": float(track['map_position'][1]),
                        "orientation": float(track['orientation']) if 'orientation' in track else None,
                        "timestamp": time.time()
                    }
                    print(f"Position: {track['map_position'][0]}, {track['map_position'][1]}")
                    # Uncomment to publish to MQTT
                    # self.publisher.publish(
                    #     f"game/player/position/{track_id}", 
                    #     json.dumps(position_data), 
                    #     qos=1
                    # )
        
        return vis_frame, tracks
    
    def run(self):
        """Main loop to receive frames and process them"""
        try:
            # Connect to server
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)  # 10 second timeout
            
            print(f"Connecting to {self.host}:{self.port}...")
            self.socket.connect((self.host, self.port))
            print("Connected successfully!")
            print("\nControls:")
            print("  'q' - Quit")
            print("  'p' - Pause/Resume processing")
            print("  's' - Save current frame")
            print()
            
            processing_enabled = True
            
            while True:
                try:
                    # Receive frame size (4 bytes)
                    size_data = self.recv_all(self.socket, 4)
                    if not size_data:
                        print("No size data received")
                        break
                    
                    frame_size = struct.unpack('!I', size_data)[0]
                    
                    # Receive frame data
                    frame_data = self.recv_all(self.socket, frame_size)
                    if not frame_data:
                        print("No frame data received")
                        break
                    
                    # Deserialize frame
                    frame = pickle.loads(frame_data)
                    
                    if frame is not None:
                        if processing_enabled:
                            # Process frame through tracking pipeline
                            vis_frame, tracks = self.process_frame(frame)
                        else:
                            # Just display raw frame
                            vis_frame = frame
                        
                        # Display frame if not headless
                        if not self.headless:
                            cv2.imshow('Live Tracking', vis_frame)
                        
                        # Process key input
                        key = cv2.waitKey(1) & 0xFF
                        
                        if key == ord('q'):
                            print("Quit requested")
                            break
                        elif key == ord('p'):
                            processing_enabled = not processing_enabled
                            print(f"Processing {'enabled' if processing_enabled else 'disabled'}")
                        elif key == ord('s'):
                            # Save current frame
                            timestamp = int(time.time())
                            filename = f"frame_{timestamp}.jpg"
                            cv2.imwrite(filename, vis_frame)
                            print(f"Frame saved as {filename}")
                    
                    else:
                        print("Received None frame")
                
                except socket.timeout:
                    print("Socket timeout")
                    break
                except Exception as e:
                    print(f"Error receiving/processing frame: {e}")
                    break
        
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.socket:
            self.socket.close()
        self.publisher.disconnect()
        if not self.headless:
            cv2.destroyAllWindows()
        print(f"Connection closed. Total frames processed: {self.frame_count}")

def main():
    # Configuration
    RPI_IP = "192.168.68.59"  # Replace with your Raspberry Pi IP
    PORT = 8080
    ROOM_INDEX = 0
    CAM_INDEX = 0
    
    # Create and run the live stream tracker
    tracker = LiveStreamTracker(
        host=RPI_IP,
        port=PORT,
        room_index=ROOM_INDEX,
        cam_index=CAM_INDEX,
        headless=False,  # Set to True to run without display
        show_fps=True
    )
    
    tracker.run()

if __name__ == "__main__":
    main()
