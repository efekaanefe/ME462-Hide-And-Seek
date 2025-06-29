#!/usr/bin/env python3
import cv2
import socket
import pickle
import struct
import threading
import time
import numpy as np

class UDPStreamServer:
    def __init__(self, host='0.0.0.0', port=8080, max_packet_size=60000, use_jpeg=False, jpeg_quality=80):
        """Initialize UDP Stream Server
        
        Args:
            host: Server host address
            port: Server port
            max_packet_size: Maximum UDP packet size (recommended: 60000 for most networks)
            use_jpeg: If True, compress frames as JPEG before sending
            jpeg_quality: JPEG compression quality (1-100, higher = better quality)
        """
        self.host = host
        self.port = port
        self.max_packet_size = max_packet_size
        self.use_jpeg = use_jpeg
        self.jpeg_quality = jpeg_quality
        self.camera = None
        self.running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.clients = set()  # Track client addresses
        self.clients_lock = threading.Lock()
        self.frame_id = 0
        
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
        frame_count = 0
        
        while self.running:
            if self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    frame_count += 1
                    
                    # Debug info every 100 frames
                    if frame_count % 100 == 0:
                        print(f"Captured {frame_count} frames, current shape: {frame.shape}")
                    
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
    
    def send_frame_fragmented(self, socket_obj, frame_data, frame_id):
        """Send frame data in fragments via UDP"""
        # Calculate number of packets needed
        data_size = len(frame_data)
        payload_size = self.max_packet_size - 8  # Reserve 8 bytes for header
        total_packets = (data_size + payload_size - 1) // payload_size  # Ceiling division
        
        if total_packets > 65535:  # Max value for 2-byte packet counter
            print(f"Frame too large: {data_size} bytes, {total_packets} packets needed")
            return False
        
        # Send each packet
        for packet_num in range(total_packets):
            start_idx = packet_num * payload_size
            end_idx = min(start_idx + payload_size, data_size)
            packet_data = frame_data[start_idx:end_idx]
            
            # Create header: frame_id (4 bytes) + packet_num (2 bytes) + total_packets (2 bytes)
            header = struct.pack('!IHH', frame_id, packet_num, total_packets)
            packet = header + packet_data
            
            # Send to all known clients
            with self.clients_lock:
                for client_addr in self.clients.copy():
                    try:
                        socket_obj.sendto(packet, client_addr)
                    except Exception as e:
                        print(f"Error sending to {client_addr}: {e}")
                        self.clients.discard(client_addr)
        
        return True
    
    def send_frame_simple(self, socket_obj, frame_data):
        """Send JPEG frame data in single UDP packet"""
        if len(frame_data) > self.max_packet_size:
            print(f"JPEG frame too large for single packet: {len(frame_data)} bytes")
            return False
        
        # Send to all known clients
        with self.clients_lock:
            for client_addr in self.clients.copy():
                try:
                    socket_obj.sendto(frame_data, client_addr)
                except Exception as e:
                    print(f"Error sending to {client_addr}: {e}")
                    self.clients.discard(client_addr)
        
        return True
    
    def handle_client_discovery(self, socket_obj):
        """Handle client discovery and registration"""
        print("Starting client discovery thread...")
        
        while self.running:
            try:
                # Listen for any incoming data (client discovery)
                socket_obj.settimeout(1.0)  # Non-blocking with timeout
                data, addr = socket_obj.recvfrom(1024)
                
                # Register new client
                with self.clients_lock:
                    if addr not in self.clients:
                        print(f"New client registered: {addr}")
                    self.clients.add(addr)
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Client discovery error: {e}")
    
    def broadcast_frames(self, socket_obj):
        """Broadcast frames to all registered clients"""
        print("Starting frame broadcast thread...")
        last_frame_time = 0
        frame_interval = 1.0 / 15  # 15 FPS
        broadcast_count = 0
        
        while self.running:
            current_time = time.time()
            
            # Control frame rate
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.01)
                continue
            
            # Debug info every 100 broadcasts
            broadcast_count += 1
            if broadcast_count % 100 == 0:
                with self.clients_lock:
                    print(f"DEBUG: Broadcast #{broadcast_count}, {len(self.clients)} clients registered")
                    if self.current_frame is not None:
                        print(f"DEBUG: Current frame shape: {self.current_frame.shape}")
                
            with self.frame_lock:
                if self.current_frame is not None:
                    frame_to_send = self.current_frame.copy()
                else:
                    frame_to_send = None
            
            if frame_to_send is not None and len(self.clients) > 0:
                try:
                    if self.use_jpeg:
                        # Compress frame as JPEG
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                        result, jpeg_data = cv2.imencode('.jpg', frame_to_send, encode_param)
                        
                        if result:
                            self.send_frame_simple(socket_obj, jpeg_data.tobytes())
                        else:
                            print("Failed to encode frame as JPEG")
                    else:
                        # Send pickled frame (fragmented if necessary)
                        frame_data = pickle.dumps(frame_to_send)
                        self.frame_id = (self.frame_id + 1) % (2**32)  # Wrap around at 32-bit limit
                        self.send_frame_fragmented(socket_obj, frame_data, self.frame_id)
                        
                    last_frame_time = current_time
                    
                except Exception as e:
                    print(f"Error broadcasting frame: {e}")
            else:
                time.sleep(0.1)
    
    def start_server(self):
        """Start the UDP server"""
        if not self.initialize_camera():
            return
            
        # Create UDP socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Increase socket buffer sizes
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024*1024)  # 1MB send buffer
        
        try:
            server_socket.bind((self.host, self.port))
            print(f"UDP Server listening on {self.host}:{self.port}")
            
            if self.use_jpeg:
                print(f"Using JPEG compression (quality: {self.jpeg_quality})")
            else:
                print(f"Using pickled frames with fragmentation (max packet: {self.max_packet_size} bytes)")
            
            self.running = True
            
            # Start frame capture thread
            capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
            capture_thread.start()
            
            # Start client discovery thread
            discovery_thread = threading.Thread(target=self.handle_client_discovery, args=(server_socket,), daemon=True)
            discovery_thread.start()
            
            # Start frame broadcast thread
            broadcast_thread = threading.Thread(target=self.broadcast_frames, args=(server_socket,), daemon=True)
            broadcast_thread.start()
            
            # Keep main thread alive
            try:
                while self.running:
                    time.sleep(10)
                    # Print status every 10 seconds
                    with self.clients_lock:
                        if len(self.clients) > 0:
                            print(f"Status: Broadcasting to {len(self.clients)} clients: {list(self.clients)}")
                        else:
                            print("Status: No clients connected, waiting for connections...")
                        
            except KeyboardInterrupt:
                print("\nShutting down server...")
                
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            self.running = False
            if self.camera:
                self.camera.release()
            server_socket.close()
            print("UDP Server stopped")


def main():
    """Main function to start the UDP stream server"""
    print("=== UDP Stream Server ===")
    print("Starting video streaming server...")
    
    # Configuration options
    config = {
        'host': '0.0.0.0',           # Listen on all interfaces
        'port': 8080,                # Default port
        'max_packet_size': 60000,    # UDP packet size limit
        'use_jpeg': False,           # Set to True for JPEG compression
        'jpeg_quality': 80           # JPEG quality (1-100)
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create and start server
    server = UDPStreamServer(**config)
    
    try:
        server.start_server()
    except Exception as e:
        print(f"Failed to start server: {e}")
    
    print("Server shutdown complete.")


if __name__ == "__main__":
    main()
