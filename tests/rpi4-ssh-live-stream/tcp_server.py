#!/usr/bin/env python3
import cv2
import socket
import pickle
import struct
import threading
import time

class TCPStreamServer:
    def __init__(self, host='0.0.0.0', port=8080):
        self.host = host
        self.port = port
        self.camera = None
        self.running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
    def initialize_camera(self):
        """Initialize camera with safe settings"""
        for i in range(3):
            print(f"Trying camera {i}...")
            self.camera = cv2.VideoCapture(i)
            
            if self.camera.isOpened():
                # Set conservative settings
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_FPS, 15)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Test capture
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    print(f"Camera {i} working: {frame.shape}, dtype: {frame.dtype}")
                    print(f"Frame stats: min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")
                    return True
                    
                self.camera.release()
        
        print("No working camera found")
        return False
    
    def capture_frames(self):
        """Capture frames from camera continuously"""
        print("Starting frame capture thread...")
        
        while self.running:
            if self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret and frame is not None:
                # Check if the frame is flattened color data
                    if len(frame.shape) == 2 and frame.shape[0] == 1 and frame.shape[1] == (640 * 480 * 3):
                        try:
                            # Reshape to (height, width, channels)
                            reshaped_frame = frame.reshape((480, 640, 3))
                            with self.frame_lock:
                                self.current_frame = reshaped_frame.copy()
                        except ValueError as e:
                            print(f"Failed to reshape frame: {e}. Original shape: {frame.shape}")
                    elif len(frame.shape) == 3 and frame.shape[2] == 3:
                        # This is the expected format, no reshaping needed
                        with self.frame_lock:
                            self.current_frame = frame.copy()
                    else:
                        print(f"Unexpected frame format: {frame.shape}")
                else:
                    print("Failed to capture frame")
                    time.sleep(0.1)
            else:
                print("Camera not available")
                time.sleep(0.5)
    
    def handle_client(self, client_socket, addr):
        """Handle individual client connections"""
        print(f"Client connected: {addr}")
        
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
                        
                time.sleep(1/15)  # Control frame rate (15 FPS)
                
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

if __name__ == "__main__":
    server = TCPStreamServer()
    server.start_server()
