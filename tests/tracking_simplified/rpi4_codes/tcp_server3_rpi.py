#!/usr/bin/env python3
import cv2
import socket
import pickle
import struct
import threading
import time
import argparse
import numpy as np
import zlib
from io import BytesIO

class TCPStreamServer:
    def __init__(self, host='0.0.0.0', port=8080, width=640, height=480, fps=30):
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
        
        # Performance settings for high resolution
        self.max_frame_size = 50 * 1024 * 1024  # 50MB max frame size
        self.compression_level = 6  # zlib compression level (1-9)
        self.use_compression = True
        self.jpeg_quality = 85  # JPEG quality for very high resolutions
        self.use_jpeg_fallback = True  # Use JPEG for frames > 4K
        
        # Buffer management
        self.frame_buffer_size = 2  # Keep only 2 frames in buffer
        self.send_timeout = 30.0  # 30 second timeout for sending large frames
        
        # Common resolution presets
        self.resolution_presets = {
            'qvga': (320, 240),
            'vga': (640, 480),
            'svga': (800, 600),
            'hd': (1280, 720),
            'fhd': (1920, 1080),
            '4k': (3840, 2160),
            'pi_hq_max': (4056, 3040),
            'pi_v2_max': (2592, 1944),
            'sci_2k': (2048, 2048),
            'sci_4k': (4096, 4096)
        }
        
    def get_supported_resolutions(self, camera):
        """Test common resolutions to find supported ones"""
        supported = []
        test_resolutions = [
            (320, 240), (640, 480), (800, 600), (1024, 768),
            (1280, 720), (1280, 960), (1600, 1200), (1920, 1080),
            (2560, 1440), (2592, 1944), (3840, 2160), (4056, 3040),
            (4096, 4096), (5120, 3840)  # Added more high-res options
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
        """Initialize camera with optimizations for high resolution"""
        for i in range(3):
            print(f"Trying camera {i}...")
            self.camera = cv2.VideoCapture(i)
            
            if self.camera.isOpened():
                # Optimize camera settings for high resolution
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
                
                # Set backend-specific optimizations
                backend = self.camera.getBackendName()
                print(f"Camera backend: {backend}")
                
                # Try different pixel formats for better performance
                if hasattr(cv2, 'CAP_PROP_FOURCC'):
                    # Try MJPG for better performance with high resolution
                    self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                
                # Set target resolution and FPS
                print(f"Setting target resolution: {self.target_width}x{self.target_height}")
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                self.camera.set(cv2.CAP_PROP_FPS, self.target_fps)
                
                # Get actual settings
                self.actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
                
                print(f"Actual settings: {self.actual_width}x{self.actual_height} @ {actual_fps:.1f} FPS")
                
                # Test capture
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    print(f"Camera {i} working!")
                    print(f"Frame shape: {frame.shape}")
                    print(f"Frame size: {frame.nbytes / (1024*1024):.2f} MB")
                    
                    # Determine if we need compression/JPEG fallback
                    frame_size_mb = frame.nbytes / (1024*1024)
                    if frame_size_mb > 25:  # >25MB frames
                        print("Large frame detected - enabling JPEG fallback")
                        self.use_jpeg_fallback = True
                    elif frame_size_mb > 10:  # >10MB frames
                        print("Medium frame detected - enabling compression")
                        self.use_compression = True
                        
                    return True
                    
                self.camera.release()
        
        print("No working camera found")
        return False
    
    def compress_frame(self, frame):
        """Compress frame data using zlib"""
        try:
            # Serialize frame first
            frame_data = pickle.dumps(frame)
            # Then compress
            compressed_data = zlib.compress(frame_data, self.compression_level)
            compression_ratio = len(frame_data) / len(compressed_data)
            
            if len(compressed_data) < len(frame_data) * 0.9:  # Only use if >10% reduction
                return compressed_data, True, compression_ratio
            else:
                return frame_data, False, 1.0
                
        except Exception as e:
            print(f"Compression failed: {e}")
            return pickle.dumps(frame), False, 1.0
    
    def encode_frame_jpeg(self, frame):
        """Encode frame as JPEG for very large frames"""
        try:
            # Encode as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            result, encoded_img = cv2.imencode('.jpg', frame, encode_param)
            
            if result:
                jpeg_data = encoded_img.tobytes()
                original_size = frame.nbytes
                jpeg_size = len(jpeg_data)
                compression_ratio = original_size / jpeg_size
                
                print(f"JPEG compression: {original_size/(1024*1024):.2f}MB -> {jpeg_size/(1024*1024):.2f}MB ({compression_ratio:.1f}x)")
                
                return jpeg_data, True
            else:
                return pickle.dumps(frame), False
                
        except Exception as e:
            print(f"JPEG encoding failed: {e}")
            return pickle.dumps(frame), False
    
    def process_frame(self, frame):
        """Process frame to handle different formats and resolutions"""
        if frame is None:
            return None
            
        # Handle different frame formats
        if len(frame.shape) == 2:
            # Grayscale
            if len(np.unique(frame)) < 10:  # Probably corrupted
                print("Suspicious grayscale frame detected")
                return None
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
        elif len(frame.shape) == 3:
            # Color frame
            if frame.shape[2] == 4:
                # RGBA to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif frame.shape[2] == 1:
                # Single channel to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # BGR frames are already in correct format
            
        else:
            print(f"Unexpected frame format: {frame.shape}")
            return None
            
        # Resize if needed
        if (frame.shape[1], frame.shape[0]) != (self.target_width, self.target_height):
            if self.target_width != self.actual_width or self.target_height != self.actual_height:
                # Use area interpolation for downscaling, cubic for upscaling
                if (self.target_width < self.actual_width or self.target_height < self.actual_height):
                    interpolation = cv2.INTER_AREA
                else:
                    interpolation = cv2.INTER_CUBIC
                    
                frame = cv2.resize(frame, (self.target_width, self.target_height), interpolation=interpolation)
                
        return frame
    
    def capture_frames(self):
        """Capture frames from camera continuously with optimizations"""
        print("Starting frame capture thread...")
        consecutive_failures = 0
        max_failures = 10
        
        while self.running:
            if self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    processed_frame = self.process_frame(frame)
                    if processed_frame is not None:
                        with self.frame_lock:
                            self.current_frame = processed_frame.copy()
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        print(f"Frame processing failed ({consecutive_failures}/{max_failures})")
                else:
                    consecutive_failures += 1
                    print(f"Failed to capture frame ({consecutive_failures}/{max_failures})")
                    
                if consecutive_failures >= max_failures:
                    print("Too many consecutive failures, restarting camera...")
                    self.camera.release()
                    time.sleep(1)
                    if self.initialize_camera():
                        consecutive_failures = 0
                    else:
                        print("Camera restart failed")
                        break
                        
            else:
                print("Camera not available")
                time.sleep(0.5)
    
    def send_large_data(self, client_socket, data, timeout=None):
        """Send large data with proper timeout and chunking"""
        if timeout:
            client_socket.settimeout(timeout)
        
        try:
            total_sent = 0
            data_len = len(data)
            chunk_size = 65536  # 64KB chunks
            
            while total_sent < data_len:
                chunk_end = min(total_sent + chunk_size, data_len)
                chunk = data[total_sent:chunk_end]
                
                sent = client_socket.send(chunk)
                if sent == 0:
                    raise RuntimeError("Socket connection broken")
                total_sent += sent
                
            return True
            
        except socket.timeout:
            print("Timeout sending large data")
            return False
        except Exception as e:
            print(f"Error sending large data: {e}")
            return False
        finally:
            if timeout:
                client_socket.settimeout(None)
    
    def handle_client(self, client_socket, addr):
        """Handle individual client connections with high-resolution optimizations"""
        print(f"Client connected: {addr}")
        
        # Increase socket buffer sizes for high resolution
        try:
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2097152)  # 2MB send buffer
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)  # 1MB receive buffer
        except:
            pass
        
        # Send resolution info to client
        try:
            resolution_info = {
                'width': self.target_width,
                'height': self.target_height,
                'fps': self.target_fps,
                'compression_enabled': self.use_compression,
                'jpeg_fallback_enabled': self.use_jpeg_fallback
            }
            info_data = pickle.dumps(resolution_info)
            client_socket.sendall(struct.pack('!I', len(info_data)))
            client_socket.sendall(info_data)
            print(f"Sent resolution info to {addr}: {resolution_info}")
        except Exception as e:
            print(f"Failed to send resolution info to {addr}: {e}")
            return
        
        frame_count = 0
        last_fps_time = time.time()
        
        try:
            while self.running:
                with self.frame_lock:
                    if self.current_frame is not None:
                        frame_to_send = self.current_frame.copy()
                    else:
                        frame_to_send = None
                
                if frame_to_send is not None:
                    try:
                        start_time = time.time()
                        
                        # Determine encoding method based on frame size
                        frame_size_mb = frame_to_send.nbytes / (1024*1024)
                        
                        if self.use_jpeg_fallback and frame_size_mb > 20:
                            # Use JPEG for very large frames
                            data, is_jpeg = self.encode_frame_jpeg(frame_to_send)
                            frame_type = 'jpeg' if is_jpeg else 'pickle'
                        else:
                            # Use compression for medium frames
                            if self.use_compression and frame_size_mb > 5:
                                data, is_compressed, ratio = self.compress_frame(frame_to_send)
                                frame_type = 'compressed' if is_compressed else 'raw'
                            else:
                                data = pickle.dumps(frame_to_send)
                                frame_type = 'raw'
                        
                        # Create header with frame info
                        header = {
                            'size': len(data),
                            'type': frame_type,
                            'original_shape': frame_to_send.shape,
                            'frame_number': frame_count
                        }
                        
                        header_data = pickle.dumps(header)
                        
                        # Send header size, header, then data
                        if not self.send_large_data(client_socket, struct.pack('!I', len(header_data)), self.send_timeout):
                            break
                        if not self.send_large_data(client_socket, header_data, self.send_timeout):
                            break
                        if not self.send_large_data(client_socket, data, self.send_timeout):
                            break
                        
                        frame_count += 1
                        
                        # Print performance stats occasionally
                        if frame_count % 30 == 0:
                            current_time = time.time()
                            elapsed = current_time - last_fps_time
                            fps = 30 / elapsed if elapsed > 0 else 0
                            encode_time = (time.time() - start_time) * 1000
                            
                            print(f"Client {addr}: Frame {frame_count}, "
                                  f"Size: {len(data)/(1024*1024):.2f}MB, "
                                  f"Type: {frame_type}, "
                                  f"Encode: {encode_time:.1f}ms, "
                                  f"FPS: {fps:.1f}")
                            
                            last_fps_time = current_time
                        
                    except Exception as e:
                        print(f"Error sending to client {addr}: {e}")
                        break
                        
                # Adaptive frame rate based on resolution
                target_delay = 1.0 / self.target_fps
                if frame_size_mb > 20:  # Reduce FPS for very large frames
                    target_delay *= 2
                elif frame_size_mb > 10:
                    target_delay *= 1.5
                    
                time.sleep(target_delay)
                
        except Exception as e:
            print(f"Client {addr} error: {e}")
        finally:
            print(f"Client {addr} disconnected (sent {frame_count} frames)")
            client_socket.close()
    
    def start_server(self):
        """Start the TCP server with high-resolution optimizations"""
        if not self.initialize_camera():
            return
            
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Increase server socket buffer
        try:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
        except:
            pass
        
        try:
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            
            print(f"TCP Server listening on {self.host}:{self.port}")
            print(f"Streaming at {self.target_width}x{self.target_height} @ {self.target_fps} FPS")
            print(f"Compression: {self.use_compression}, JPEG fallback: {self.use_jpeg_fallback}")
            
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
    parser = argparse.ArgumentParser(description='High-Resolution TCP Video Stream Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to (default: 8080)')
    parser.add_argument('--width', type=int, default=640, help='Frame width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Frame height (default: 480)')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS (default: 30)')
    parser.add_argument('--preset', help='Use resolution preset (qvga, vga, svga, hd, fhd, 4k, etc.)')
    parser.add_argument('--list-presets', action='store_true', help='List available resolution presets')
    parser.add_argument('--no-compression', action='store_true', help='Disable frame compression')
    parser.add_argument('--no-jpeg-fallback', action='store_true', help='Disable JPEG fallback for large frames')
    parser.add_argument('--jpeg-quality', type=int, default=85, help='JPEG quality (1-100, default: 85)')
    parser.add_argument('--compression-level', type=int, default=6, help='Compression level (1-9, default: 6)')
    
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
            'pi_v2_max': (2592, 1944),
            'sci_2k': (2048, 2048),
            'sci_4k': (4096, 4096)
        }
        for name, (w, h) in presets.items():
            print(f"  {name}: {w}x{h}")
        return
    
    # Create server instance
    server = TCPStreamServer(args.host, args.port, args.width, args.height, args.fps)
    
    # Apply command line options
    if args.no_compression:
        server.use_compression = False
    if args.no_jpeg_fallback:
        server.use_jpeg_fallback = False
    server.jpeg_quality = args.jpeg_quality
    server.compression_level = args.compression_level
    
    # Use preset if specified
    if args.preset:
        if not server.set_resolution_from_preset(args.preset):
            print(f"Unknown preset: {args.preset}")
            print("Use --list-presets to see available options")
            return
    
    server.start_server()

if __name__ == "__main__":
    main()