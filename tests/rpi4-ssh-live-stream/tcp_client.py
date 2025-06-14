#!/usr/bin/env python3
import cv2
import socket
import pickle
import struct
import numpy as np
import time

class TCPStreamClient:
    def __init__(self, host, port=8080):
        self.host = host
        self.port = port
        self.socket = None
        
    def recv_all(self, sock, n):
        """Receive exactly n bytes"""
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data
    
    def connect_and_receive(self):
        """Connect to server and receive frames"""
        try:
            # Connect to server
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)  # 10 second timeout
            
            print(f"Connecting to {self.host}:{self.port}...")
            self.socket.connect((self.host, self.port))
            print("Connected successfully!")
            
            frame_count = 0
            
            while True:
                try:
                    # Receive frame size (4 bytes)
                    size_data = self.recv_all(self.socket, 4)
                    if not size_data:
                        print("No size data received")
                        break
                        
                    frame_size = struct.unpack('!I', size_data)[0]
                    print(f"Expecting frame of size: {frame_size} bytes")
                    
                    # Receive frame data
                    frame_data = self.recv_all(self.socket, frame_size)
                    if not frame_data:
                        print("No frame data received")
                        break
                    
                    # Deserialize frame
                    frame = pickle.loads(frame_data)
                    
                    if frame is not None:
                        frame_count += 1
                        
                        # Print frame info for first few frames
                        if frame_count <= 5:
                            print(f"Frame {frame_count}: shape={frame.shape}, dtype={frame.dtype}")
                            print(f"Frame stats: min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")
                        
                        # Display frame
                        cv2.imshow('TCP Camera Stream', frame)
                        
                        # Process key input
                        key = cv2.waitKey(1) & 0xFF
                        
                        # Press 'g' for grayscale
                        if key == ord('g'):
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            cv2.imshow('Grayscale', gray)
                        
                        # Press 'e' for edge detection
                        elif key == ord('e'):
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            edges = cv2.Canny(gray, 50, 150)
                            cv2.imshow('Edges', edges)
                        
                        # Press 'q' to quit
                        elif key == ord('q'):
                            print("Quit requested")
                            break
                    
                    else:
                        print("Received None frame")
                        
                except socket.timeout:
                    print("Socket timeout")
                    break
                except Exception as e:
                    print(f"Error receiving frame: {e}")
                    break
                    
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            if self.socket:
                self.socket.close()
            cv2.destroyAllWindows()
            print(f"Connection closed. Total frames received: {frame_count}")

def main():
    # Replace with your RPi's IP address
    RPI_IP = "192.168.68.59"
    
    print("TCP Camera Stream Client")
    print("Controls:")
    print("  'g' - Toggle grayscale view")
    print("  'e' - Toggle edge detection")
    print("  'q' - Quit")
    print()
    
    client = TCPStreamClient(RPI_IP)
    client.connect_and_receive()

if __name__ == "__main__":
    main()
