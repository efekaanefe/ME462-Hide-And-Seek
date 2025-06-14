#!/usr/bin/env python3
import socket
import pickle
import struct
import cv2
import threading
import time

class UDPClient:
    """Modular UDP client for receiving live video frames"""
    
    def __init__(self, host, port=8080, timeout=5, buffer_size=65536):
        """Initialize UDP client
        
        Args:
            host: Server IP address (e.g., Raspberry Pi IP)
            port: Server port
            timeout: Socket timeout in seconds
            buffer_size: UDP receive buffer size
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.buffer_size = buffer_size
        self.socket = None
        self.connected = False
        self.frame_buffer = {}  # For handling fragmented frames
        self.registration_thread = None
        self.stop_registration = False
        
    def connect(self):
        """Initialize UDP socket and register with server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(self.timeout)
            
            # Increase receive buffer size for better performance
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.buffer_size * 4)
            
            print(f"UDP client initialized for {self.host}:{self.port}")
            self.connected = True
            
            # Start registration thread
            self.stop_registration = False
            self.registration_thread = threading.Thread(target=self._register_with_server, daemon=True)
            self.registration_thread.start()
            
            return True
            
        except Exception as e:
            print(f"UDP initialization error: {e}")
            self.connected = False
            return False
    
    def _register_with_server(self):
        """Send periodic registration messages to server"""
        while not self.stop_registration and self.connected:
            try:
                # Send a simple discovery message
                discovery_msg = b"CLIENT_DISCOVERY"
                self.socket.sendto(discovery_msg, (self.host, self.port))
                print(f"Sent registration to {self.host}:{self.port}")
                time.sleep(5)  # Re-register every 5 seconds
            except Exception as e:
                print(f"Registration error: {e}")
                time.sleep(1)
    
    def get_frame(self):
        """Get next frame from UDP stream
        
        Returns:
            numpy.ndarray: Frame image, or None if error/timeout
        """
        if not self.connected:
            return None
            
        try:
            while True:
                # Receive UDP packet
                data, addr = self.socket.recvfrom(self.buffer_size)
                
                # Skip discovery messages (our own registration messages echoed back)
                if data == b"CLIENT_DISCOVERY":
                    continue
                
                # Check if this is a single-packet frame or multi-packet frame
                if len(data) < 8:  # Not enough data for header
                    continue
                    
                # Parse header: frame_id (4 bytes) + packet_num (2 bytes) + total_packets (2 bytes)
                try:
                    frame_id = struct.unpack('!I', data[:4])[0]
                    packet_num = struct.unpack('!H', data[4:6])[0]
                    total_packets = struct.unpack('!H', data[6:8])[0]
                except struct.error:
                    # Not a frame packet, skip
                    continue
                
                frame_data = data[8:]  # Actual frame data
                
                # Handle single packet frame
                if total_packets == 1:
                    try:
                        frame = pickle.loads(frame_data)
                        return frame
                    except Exception as e:
                        print(f"Error deserializing single packet frame: {e}")
                        continue
                
                # Handle multi-packet frame
                if frame_id not in self.frame_buffer:
                    self.frame_buffer[frame_id] = {}
                
                self.frame_buffer[frame_id][packet_num] = frame_data
                
                # Check if we have all packets for this frame
                if len(self.frame_buffer[frame_id]) == total_packets:
                    # Reconstruct frame
                    complete_data = b''
                    for i in range(total_packets):
                        if i in self.frame_buffer[frame_id]:
                            complete_data += self.frame_buffer[frame_id][i]
                        else:
                            # Missing packet, discard frame
                            del self.frame_buffer[frame_id]
                            break
                    else:
                        # All packets received, deserialize frame
                        try:
                            frame = pickle.loads(complete_data)
                            del self.frame_buffer[frame_id]
                            
                            # Clean up old incomplete frames (keep only last 10 frame IDs)
                            if len(self.frame_buffer) > 10:
                                oldest_frame_id = min(self.frame_buffer.keys())
                                del self.frame_buffer[oldest_frame_id]
                            
                            return frame
                        except Exception as e:
                            print(f"Error deserializing multi-packet frame: {e}")
                            del self.frame_buffer[frame_id]
                            continue
                            
        except socket.timeout:
            print("UDP timeout - no frame received")
            return None
        except Exception as e:
            print(f"Error getting UDP frame: {e}")
            return None
    
    def is_connected(self):
        """Check if client is initialized"""
        return self.connected
    
    def disconnect(self):
        """Close UDP socket"""
        self.stop_registration = True
        if self.registration_thread:
            self.registration_thread.join(timeout=1)
            
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False
        self.frame_buffer.clear()
        print("UDP client disconnected")



