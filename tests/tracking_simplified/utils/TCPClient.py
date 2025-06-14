
#!/usr/bin/env python3
import socket
import pickle
import struct

class TCPClient:
    """Modular TCP client for receiving live video frames"""
    
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
        
    def connect(self):
        """Connect to the TCP server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            
            print(f"Connecting to {self.host}:{self.port}...")
            self.socket.connect((self.host, self.port))
            self.connected = True
            print("Connected successfully!")
            return True
            
        except Exception as e:
            print(f"Connection error: {e}")
            self.connected = False
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
            return frame
            
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None
    
    def is_connected(self):
        """Check if client is connected"""
        return self.connected
    
    def disconnect(self):
        """Disconnect from server"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False
        print("TCP client disconnected")
