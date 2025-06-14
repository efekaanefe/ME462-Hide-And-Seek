#!/usr/bin/env python3
import cv2
import numpy as np
import requests
import time

# Replace with your RPi's IP address
RPI_IP = "192.168.68.59"  # Your RPi's IP
STREAM_URL = f"http://{RPI_IP}:5000/video_feed"

def test_connection():
    """Test if the stream URL is accessible"""
    try:
        response = requests.get(f"http://{RPI_IP}:5000/status", timeout=5)
        print(f"Server status: {response.text}")
        return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False

def main():
    print(f"Testing connection to: {RPI_IP}")
    
    if not test_connection():
        print("Cannot connect to RPi server. Check:")
        print("1. RPi server is running")
        print("2. IP address is correct")
        print("3. Both devices are on same network")
        return
    
    print(f"Connecting to stream: {STREAM_URL}")
    
    # Configure OpenCV capture with specific parameters
    cap = cv2.VideoCapture(STREAM_URL)
    
    # Set buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("Error: Could not open video stream")
        print("Try opening this URL in a web browser:", f"http://{RPI_IP}:5000")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if ret and frame is not None:
            frame_count += 1
            
            # Check if frame looks corrupted (mostly black/white vertical lines)
            if frame_count < 10:  # Check first few frames
                gray_check = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_val = np.mean(gray_check)
                std_val = np.std(gray_check)
                print(f"Frame {frame_count}: mean={mean_val:.1f}, std={std_val:.1f}, shape={frame.shape}")
                
                if std_val < 10:  # Very low variation = likely corrupted
                    print("Warning: Frame appears corrupted (low variation)")
                    continue
            
            # Resize if frame is too large for display
            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            
            # Display original frame only (remove dual window)
            cv2.imshow('RPi Camera Stream', frame)
            
            # Optional: Add some basic processing
            if cv2.waitKey(1) & 0xFF == ord('g'):  # Press 'g' for grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('Grayscale', gray)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        else:
            print("No frame received, retrying...")
            time.sleep(0.5)
            
            # Try to reconnect
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(STREAM_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    cap.release()
    cv2.destroyAllWindows()
    print("Stream closed")

if __name__ == "__main__":
    main()
