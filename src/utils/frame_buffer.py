import time
import threading
import queue
from collections import deque


class FrameBuffer:
    """Thread-safe frame buffer that can work as either queue (FIFO) or stack (LIFO)"""
    
    def __init__(self, maxsize=30, use_queue=True):
        """
        Args:
            maxsize: Maximum number of frames to buffer
            use_queue: If True, use FIFO (queue), if False, use LIFO (stack)
        """
        self.maxsize = maxsize
        self.use_queue = use_queue
        self.lock = threading.Lock()
        
        if use_queue:
            self.buffer = queue.Queue(maxsize=maxsize)
        else:
            self.buffer = deque(maxlen=maxsize)
    
    def put(self, frame):
        """Add frame to buffer"""
        with self.lock:
            if self.use_queue:
                try:
                    self.buffer.put_nowait(frame)
                except queue.Full:
                    # Remove oldest frame and add new one
                    try:
                        self.buffer.get_nowait()
                        self.buffer.put_nowait(frame)
                    except queue.Empty:
                        pass
            else:
                # For stack (deque), just append - maxlen handles overflow
                self.buffer.append(frame)
    
    def get(self):
        """Get frame from buffer"""
        with self.lock:
            if self.use_queue:
                try:
                    return self.buffer.get_nowait()
                except queue.Empty:
                    return None
            else:
                try:
                    return self.buffer.pop()  # LIFO - get most recent
                except IndexError:
                    return None
    
    def empty(self):
        """Check if buffer is empty"""
        with self.lock:
            if self.use_queue:
                return self.buffer.empty()
            else:
                return len(self.buffer) == 0
    
    def size(self):
        """Get current buffer size"""
        with self.lock:
            if self.use_queue:
                return self.buffer.qsize()
            else:
                return len(self.buffer)


def frame_reader_thread(tcp_client, frame_buffer, stop_event):
    """Thread function to continuously read frames from TCP client"""
    print("Frame reader thread started")
    
    while not stop_event.is_set():
        if not tcp_client.is_connected():
            print("TCP connection lost, attempting to reconnect...")
            time.sleep(1)
            if not tcp_client.connect():
                print("Failed to reconnect, stopping frame reader")
                break
            continue
            
        frame = tcp_client.get_frame()
        if frame is not None:
            frame_buffer.put(frame)
        else:
            # Small delay to prevent busy waiting
            time.sleep(0.001)
    
    print("Frame reader thread stopped")
