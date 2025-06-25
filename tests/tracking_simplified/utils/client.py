#!/usr/bin/env python3
"""
Camera streaming client for laptop
Run this script on your laptop to view the RPi4 camera stream
"""

import cv2
import socket
import struct
import pickle
import numpy as np

class CameraClient:
    def __init__(self, server_ip='192.168.0.135', server_port=8080):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_socket = None
        self.connected = False
        
    def connect_to_server(self):
        """Connect to the camera streaming server"""
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.server_ip, self.server_port))
            self.connected = True
            print(f"Connected to camera server at {self.server_ip}:{self.server_port}")
            return True
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return False
    
    def receive_frame(self):
        """Receive a single frame from the server"""
        try:
            # First, receive the size of the incoming data
            packed_size = self.client_socket.recv(struct.calcsize("L"))
            if not packed_size:
                return None
            
            data_size = struct.unpack("L", packed_size)[0]
            
            # Receive the actual frame data
            data = b""
            while len(data) < data_size:
                packet = self.client_socket.recv(data_size - len(data))
                if not packet:
                    return None
                data += packet
            
            # Deserialize the frame
            encoded_frame = pickle.loads(data)
            
            # Decode the JPEG frame
            frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
            return frame
            
        except Exception as e:
            print(f"Error receiving frame: {e}")
            return None
    
    def start_viewing(self):
        """Start viewing the camera stream"""
        if not self.connect_to_server():
            return
        
        print("Starting camera stream viewer. Press 'q' to quit.")
        
        try:
            while self.connected:
                frame = self.receive_frame()
                
                if frame is None:
                    print("Lost connection to server")
                    break
                
                # Add connection info overlay
                cv2.putText(frame, f"RPi4 Camera - {self.server_ip}:{self.server_port}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display the frame
                cv2.imshow('RPi4 Camera Stream', frame)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit key pressed")
                    break
                    
        except KeyboardInterrupt:
            print("\nStream interrupted by user")
        except Exception as e:
            print(f"Streaming error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.connected = False
        
        if self.client_socket:
            self.client_socket.close()
            
        cv2.destroyAllWindows()
        print("Client cleanup completed")

if __name__ == "__main__":
    # Create and start the camera client
    client = CameraClient(server_ip='192.168.0.135', server_port=8080)
    
    try:
        client.start_viewing()
    except KeyboardInterrupt:
        print("\nClient stopped by user")
    except Exception as e:
        print(f"Client error: {e}")