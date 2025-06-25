
#!/usr/bin/env python3
import socket
import pickle
import struct

#!/usr/bin/env python3
import socket
import pickle
import struct
import cv2
import numpy as np

class TCPClient:
    """Modular TCP client for receiving live video frames with resolution support"""
    
    def __init__(self, host, port=8080, timeout=10):
        """Initialize TCP client
        
        Args:
            host: Server IP address (e.g., Raspberry Pi IP)
            port: Server port
            timeout: Socket timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.connected = False
        self.stream_info = None
        self.frame_count = 0
        
    def connect(self):
        """Connect to the TCP server and receive stream information"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            print(f"Connecting to {self.host}:{self.port}...")
            self.socket.connect((self.host, self.port))
            
            # Receive stream information first
            if not self._receive_stream_info():
                print("Failed to receive stream information")
                return False
                
            self.connected = True
            print("Connected successfully!")
            print(f"Stream: {self.stream_info['width']}x{self.stream_info['height']} @ {self.stream_info['fps']} FPS")
            return True
            
        except Exception as e:
            print(f"Connection error: {e}")
            self.connected = False
            return False
    
    def _receive_stream_info(self):
        """Receive stream information from server"""
        try:
            # Receive info size (4 bytes)
            size_data = self.recv_all(4)
            if not size_data:
                return False
                
            info_size = struct.unpack('!I', size_data)[0]
            
            # Receive info data
            info_data = self.recv_all(info_size)
            if not info_data:
                return False
                
            # Deserialize stream info
            self.stream_info = pickle.loads(info_data)
            return True
            
        except Exception as e:
            print(f"Error receiving stream info: {e}")
            return False
    
    def recv_all(self, n):
        """Receive exactly n bytes"""
        data = b''
        while len(data) < n:
            try:
                packet = self.socket.recv(n - len(data))
                if not packet:
                    return None
                data += packet
            except socket.timeout:
                print("Socket timeout while receiving data")
                return None
            except Exception as e:
                print(f"Error receiving data: {e}")
                return None
        return data
    
    def get_frame(self):
        """Get next frame from stream
        
        Returns:
            numpy.ndarray: Frame image, or None if error/timeout
        """
        if not self.connected:
            return None
            
        try:
            # Receive frame size (4 bytes)
            size_data = self.recv_all(4)
            if not size_data:
                return None
                
            frame_size = struct.unpack('!I', size_data)[0]
            
            # Receive frame data
            frame_data = self.recv_all(frame_size)
            if not frame_data:
                return None
                
            # Deserialize frame
            frame = pickle.loads(frame_data)
            
            # Validate frame
            if not self._validate_frame(frame):
                return None
                
            self.frame_count += 1
            if self.frame_count % 30 == 0:  # Print every 30 frames
                print(f"Frame {self.frame_count}: {frame.shape}, dtype: {frame.dtype}")
                
            return frame
            
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None
    
    def _validate_frame(self, frame):
        """Validate received frame"""
        if frame is None:
            print("Received None frame")
            return False
            
        if not isinstance(frame, np.ndarray):
            print(f"Frame is not numpy array: {type(frame)}")
            return False
            
        if len(frame.shape) not in [2, 3]:
            print(f"Invalid frame dimensions: {frame.shape}")
            return False
            
        if len(frame.shape) == 3 and frame.shape[2] not in [1, 3, 4]:
            print(f"Invalid number of channels: {frame.shape[2]}")
            return False
            
        # Check if frame size matches expected (if we have stream info)
        if self.stream_info:
            expected_height = self.stream_info['height']
            expected_width = self.stream_info['width']
            
            if len(frame.shape) >= 2:
                actual_height, actual_width = frame.shape[:2]
                if actual_height != expected_height or actual_width != expected_width:
                    print(f"Frame size mismatch: got {actual_width}x{actual_height}, expected {expected_width}x{expected_height}")
                    # Don't return False here, just warn - server might be doing resize
                    
        return True
    
    def get_stream_info(self):
        """Get stream information
        
        Returns:
            dict: Stream info with width, height, fps, or None if not connected
        """
        return self.stream_info.copy() if self.stream_info else None
    
    def is_connected(self):
        """Check if client is connected"""
        return self.connected
    
    def get_frame_count(self):
        """Get number of frames received"""
        return self.frame_count
    
    def reset_frame_count(self):
        """Reset frame counter"""
        self.frame_count = 0
    
    def disconnect(self):
        """Disconnect from server"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False
        self.stream_info = None
        print("TCP client disconnected")
