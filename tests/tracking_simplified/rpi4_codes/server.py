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
        self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            raise Exception("Could not open camera")
        
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera initialized successfully")
        
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
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Encode frame as JPEG for compression
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
                
                if not result:
                    continue
                
                # Serialize the frame
                data = pickle.dumps(encoded_frame)
                
                # Send frame size first, then the frame data
                try:
                    # Pack the size of the data and send it
                    size = struct.pack("L", len(data))
                    client_socket.sendall(size + data)
                except (ConnectionResetError, BrokenPipeError):
                    print(f"Client {client_address} disconnected")
                    break
                
                # Small delay to control frame rate
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            print(f"Error handling client {client_address}: {e}")
        finally:
            client_socket.close()
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
    # Create and start the camera streamer
    streamer = CameraStreamer(host='0.0.0.0', port=8080, camera_index=0)
    
    try:
        streamer.start_streaming()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")