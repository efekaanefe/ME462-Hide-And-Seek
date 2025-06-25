#!/usr/bin/env python3
"""
Camera streaming server for Raspberry Pi 4
Run this script on your RPi4 (192.168.0.135)
"""

import cv2
import socket
import struct
import pickle
import threading
import time

class CameraStreamer:
    def __init__(self, host='0.0.0.0', port=8080, camera_index=0):
        self.host = host
        self.port = port
        self.camera_index = camera_index
        self.server_socket = None
        self.camera = None
        self.clients = []
        self.running = False
        
    def initialize_camera(self):
        """Initialize the camera with optimal settings"""
        # Try different camera indices and backends
        camera_indices = [0, 1, 2, -1]  # -1 for any available camera
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        
        for backend in backends:
            for idx in camera_indices:
                print(f"Trying camera index {idx} with backend {backend}")
                self.camera = cv2.VideoCapture(idx, backend)
                
                if self.camera.isOpened():
                    # Test if we can actually read a frame
                    ret, test_frame = self.camera.read()
                    if ret and test_frame is not None:
                        print(f"Successfully opened camera {idx} with backend {backend}")
                        break
                    else:
                        self.camera.release()
                        self.camera = None
                else:
                    if self.camera:
                        self.camera.release()
                        self.camera = None
            
            if self.camera and self.camera.isOpened():
                break
        
        if not self.camera or not self.camera.isOpened():
            raise Exception("Could not open any camera. Check if camera is connected and not in use.")
        
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 15)  # Reduced FPS for stability
        
        # Get actual camera properties
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized successfully")
        print(f"Resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
        
    def setup_server(self):
        """Setup the server socket"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"Server listening on {self.host}:{self.port}")
        
    def handle_client(self, client_socket, client_address):
        """Handle individual client connections"""
        print(f"Client connected from {client_address}")
        
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    print("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Validate frame dimensions
                if frame.shape[0] == 0 or frame.shape[1] == 0:
                    print("Invalid frame dimensions")
                    continue
                
                # Resize frame if it's too large (safety check)
                height, width = frame.shape[:2]
                if width > 1920 or height > 1080:
                    frame = cv2.resize(frame, (640, 480))
                    print(f"Resized large frame from {width}x{height} to 640x480")
                
                # Encode frame as JPEG with error handling
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                try:
                    result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
                    
                    if not result or encoded_frame is None:
                        print("Failed to encode frame")
                        continue
                        
                except Exception as encode_error:
                    print(f"Encoding error: {encode_error}")
                    continue
                
                # Serialize the frame
                try:
                    data = pickle.dumps(encoded_frame)
                except Exception as pickle_error:
                    print(f"Pickle error: {pickle_error}")
                    continue
                
                # Send frame size first, then the frame data
                try:
                    # Pack the size of the data and send it
                    size = struct.pack("L", len(data))
                    client_socket.sendall(size + data)
                except (ConnectionResetError, BrokenPipeError, OSError) as network_error:
                    print(f"Client {client_address} disconnected: {network_error}")
                    break
                except Exception as send_error:
                    print(f"Send error for client {client_address}: {send_error}")
                    break
                
                # Control frame rate
                time.sleep(0.067)  # ~15 FPS
                
        except Exception as e:
            print(f"Error handling client {client_address}: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass
            if client_socket in self.clients:
                self.clients.remove(client_socket)
            print(f"Client {client_address} connection closed")
    
    def start_streaming(self):
        """Start the camera streaming server"""
        try:
            self.initialize_camera()
            self.setup_server()
            self.running = True
            
            print("Camera streaming server started. Press Ctrl+C to stop.")
            
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    self.clients.append(client_socket)
                    
                    # Handle each client in a separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except KeyboardInterrupt:
                    print("\nShutting down server...")
                    break
                except Exception as e:
                    print(f"Server error: {e}")
                    
        except Exception as e:
            print(f"Failed to start server: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        # Close all client connections
        for client in self.clients:
            try:
                client.close()
            except:
                pass
        
        # Close server socket
        if self.server_socket:
            self.server_socket.close()
            
        # Release camera
        if self.camera:
            self.camera.release()
            
        print("Server cleanup completed")

if __name__ == "__main__":
    print("RPi4 Camera Streaming Server")
    print("============================")
    
    # Check available cameras first
    print("Checking available cameras...")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i}: Available ({frame.shape[1]}x{frame.shape[0]})")
            cap.release()
        else:
            print(f"Camera {i}: Not available")
    
    print("\nStarting camera streamer...")
    
    # Create and start the camera streamer
    streamer = CameraStreamer(host='0.0.0.0', port=8080, camera_index=0)
    
    try:
        streamer.start_streaming()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
        print("Possible solutions:")
        print("1. Check if camera is connected and working")
        print("2. Try running: sudo usermod -a -G video $USER")
        print("3. Reboot and try again")
        print("4. Check camera permissions: ls -l /dev/video*")