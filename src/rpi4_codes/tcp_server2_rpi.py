#!/usr/bin/env python3
import cv2
import socket
import pickle
import struct
import threading
import time
import argparse

class TCPStreamServer:
    def __init__(self, host='192.168.0.135', port=8080, width=640, height=480, fps=30):
        self.host = host
        self.port = port
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.camera = None
        self.running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.actual_width = None
        self.actual_height = None
        
        # Common resolution presets
        self.resolution_presets = {
            'qvga': (320, 240),
            'vga': (640, 480),
            'svga': (800, 600),
            'hd': (1280, 720),
            'fhd': (1920, 1080),
            '4k': (3840, 2160),
            'pi_hq_max': (4056, 3040),
            'pi_v2_max': (2592, 1944)
        }
        
    def get_supported_resolutions(self, camera):
        """Test common resolutions to find supported ones"""
        supported = []
        test_resolutions = [
            (320, 240), (640, 480), (800, 600), (1024, 768),
            (1280, 720), (1280, 960), (1600, 1200), (1920, 1080),
            (2560, 1440), (2592, 1944), (3840, 2160), (4056, 3040)
        ]
        
        print("Testing supported resolutions...")
        for width, height in test_resolutions:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_w == width and actual_h == height:
                supported.append((width, height))
                print(f"  ✓ {width}x{height}")
            else:
                print(f"  ✗ {width}x{height} -> {actual_w}x{actual_h}")
                
        return supported
        
    def set_resolution_from_preset(self, preset_name):
        """Set resolution from preset name"""
        if preset_name.lower() in self.resolution_presets:
            self.target_width, self.target_height = self.resolution_presets[preset_name.lower()]
            print(f"Using preset '{preset_name}': {self.target_width}x{self.target_height}")
            return True
        return False
        
    def initialize_camera(self):
        """Initialize camera with flexible resolution support"""
        print(f"Trying camera {0}...")
        self.camera = cv2.VideoCapture(0)
        
        if self.camera.isOpened():
            # First, get supported resolutions
            supported_resolutions = self.get_supported_resolutions(self.camera)
            
            if not supported_resolutions:
                print(f"Camera 0: No supported resolutions found")
                self.camera.release()
            
            # Try to set target resolution
            print(f"Setting target resolution: {self.target_width}x{self.target_height}")
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Get actual resolution set
            self.actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            print(f"Actual settings: {self.actual_width}x{self.actual_height} @ {actual_fps:.1f} FPS")
            
            # Test capture with actual resolution
            ret, frame = self.camera.read()
            if ret and frame is not None:
                print(f"Camera {0} working!")
                print(f"Frame shape: {frame.shape}")
                print(f"Frame dtype: {frame.dtype}")
                print(f"Frame stats: min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")
                
                # Verify frame dimensions match expected
                if len(frame.shape) >= 2:
                    frame_h, frame_w = frame.shape[:2]
                    if frame_h != self.actual_height or frame_w != self.actual_width:
                        print(f"Warning: Frame size {frame_w}x{frame_h} doesn't match camera settings")
                        
                return True
                    
                self.camera.release()
        
        print("No working camera found")
        return False
    
    def process_frame(self, frame):
        if frame is None:
            return None

        # Handle different frame formats and ensure consistent BGR format first
        if len(frame.shape) == 2:
            if frame.shape[0] == 1:
                total_pixels = frame.shape[1]
                expected_bgr_pixels = self.actual_width * self.actual_height * 3
                expected_gray_pixels = self.actual_width * self.actual_height

                if total_pixels >= expected_bgr_pixels:
                    try:
                        frame = frame[:, :expected_bgr_pixels]
                        frame = frame.reshape((self.actual_height, self.actual_width, 3))
                        # Assume this is already in BGR format from camera
                    except ValueError:
                        return None
                elif total_pixels >= expected_gray_pixels:
                    try:
                        frame = frame[:, :expected_gray_pixels]
                        frame = frame.reshape((self.actual_height, self.actual_width))
                        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    except ValueError:
                        return None
                else:
                    return None
            else:
                print()
                # Grayscale to BGR
                # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        elif len(frame.shape) == 3:
            if frame.shape[2] == 4:
                # RGBA to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif frame.shape[2] == 1:
                print()
                # Single channel to BGR
                # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # If frame.shape[2] == 3, assume it's already BGR from OpenCV camera
        else:
            return None

        # Resize if needed
        if (frame.shape[1], frame.shape[0]) != (self.target_width, self.target_height):
            if self.target_width != self.actual_width or self.target_height != self.actual_height:
                print(f"Resized from {self.actual_width}x{self.actual_height} to {self.target_width}x{self.target_height}")
                frame = cv2.resize(frame, (self.target_width, self.target_height))

        # Convert BGR to RGB for consistent output regardless of resolution
        if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    
    def capture_frames(self):
        """Capture frames from camera continuously"""
        print("Starting frame capture thread...")
        
        while self.running:
            if self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    processed_frame = self.process_frame(frame)
                    if processed_frame is not None:
                        with self.frame_lock:
                            self.current_frame = processed_frame.copy()
                    else:
                        print("Frame processing failed")
                else:
                    print("Failed to capture frame")
                    time.sleep(0.1)
            else:
                print("Camera not available")
                time.sleep(0.5)
    
    def handle_client(self, client_socket, addr):
        """Handle individual client connections"""
        print(f"Client connected: {addr}")
        
        # Send resolution info to client
        try:
            resolution_info = {
                'width': self.target_width,
                'height': self.target_height,
                'fps': self.target_fps
            }
            info_data = pickle.dumps(resolution_info)
            client_socket.sendall(struct.pack('!I', len(info_data)))
            client_socket.sendall(info_data)
            print(f"Sent resolution info to {addr}: {resolution_info}")
        except Exception as e:
            print(f"Failed to send resolution info to {addr}: {e}")
        
        try:
            while self.running:
                with self.frame_lock:
                    if self.current_frame is not None:
                        frame_to_send = self.current_frame.copy()
                    else:
                        frame_to_send = None
                
                if frame_to_send is not None:
                    try:
                        # Serialize frame
                        data = pickle.dumps(frame_to_send)
                        size = len(data)
                        
                        # Send size first (4 bytes), then data
                        client_socket.sendall(struct.pack('!I', size))
                        client_socket.sendall(data)
                        
                    except Exception as e:
                        print(f"Error sending to client {addr}: {e}")
                        break
                        
                time.sleep(1/self.target_fps)  # Control frame rate
                
        except Exception as e:
            print(f"Client {addr} error: {e}")
        finally:
            print(f"Client {addr} disconnected")
            client_socket.close()
    
    def start_server(self):
        """Start the TCP server"""
        if not self.initialize_camera():
            return
            
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            
            print(f"TCP Server listening on {self.host}:{self.port}")
            print(f"Streaming at {self.target_width}x{self.target_height} @ {self.target_fps} FPS")
            self.running = True
            
            # Start frame capture thread
            capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
            capture_thread.start()
            
            while self.running:
                try:
                    client_socket, addr = server_socket.accept()
                    
                    # Handle each client in a separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client, 
                        args=(client_socket, addr),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.error as e:
                    if self.running:
                        print(f"Socket error: {e}")
                        
        except KeyboardInterrupt:
            print("\nShutting down server...")
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            self.running = False
            if self.camera:
                self.camera.release()
            server_socket.close()
            print("Server stopped")

def main():
    parser = argparse.ArgumentParser(description='TCP Video Stream Server with flexible resolution support')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to (default: 8080)')
    parser.add_argument('--width', type=int, default=640, help='Frame width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Frame height (default: 480)')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS (default: 30)')
    parser.add_argument('--preset', help='Use resolution preset (qvga, vga, svga, hd, fhd, 4k, pi_hq_max, pi_v2_max)')
    parser.add_argument('--list-presets', action='store_true', help='List available resolution presets')
    
    args = parser.parse_args()
    
    if args.list_presets:
        print("Available resolution presets:")
        presets = {
            'qvga': (320, 240),
            'vga': (640, 480),
            'svga': (800, 600),
            'hd': (1280, 720),
            'fhd': (1920, 1080),
            '4k': (3840, 2160),
            'pi_hq_max': (4056, 3040),
            'pi_v2_max': (2592, 1944)
        }
        for name, (w, h) in presets.items():
            print(f"  {name}: {w}x{h}")
        return
    
    # Create server instance
    server = TCPStreamServer(args.host, args.port, args.width, args.height, args.fps)
    
    # Use preset if specified
    if args.preset:
        if not server.set_resolution_from_preset(args.preset):
            print(f"Unknown preset: {args.preset}")
            print("Use --list-presets to see available options")
            return
    
    server.start_server()

if __name__ == "__main__":
    main()