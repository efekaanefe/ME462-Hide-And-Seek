import cv2
import numpy as np
import requests
from threading import Thread
import time

class VideoStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self
        
    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()
            
    def read(self):
        return self.frame
        
    def stop(self):
        self.stopped = True
        self.stream.release()

# Replace with your RPi's IP address
RPI_IP = "192.168.68.59"  # Change this to your RPi's actual IP
STREAM_URL = f"http://{RPI_IP}:5000/video_feed"

def main():
    print(f"Connecting to stream: {STREAM_URL}")
    
    # Initialize video stream
    vs = VideoStream(STREAM_URL).start()
    time.sleep(2.0)  # Allow camera to warm up
    
    while True:
        frame = vs.read()
        
        if frame is not None:
            # Your OpenCV processing here
            # Example: Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Example: Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Display frames
            cv2.imshow('Original', frame)
            cv2.imshow('Edges', edges)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("No frame received")
            time.sleep(0.1)
    
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
