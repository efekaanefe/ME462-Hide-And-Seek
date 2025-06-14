#!/usr/bin/env python3
import cv2
from flask import Flask, Response
import threading
import time
import numpy as np

app = Flask(__name__)
camera = None
camera_lock = threading.Lock()

def initialize_camera():
    global camera
    # Try different camera indices and backends
    for i in range(3):
        print(f"Trying camera index {i}")
        camera = cv2.VideoCapture(i)
        
        if camera.isOpened():
            # Set reasonable resolution
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for stability
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer
            
            # Test frame capture
            ret, test_frame = camera.read()
            if ret and test_frame is not None:
                h, w = test_frame.shape[:2]
                print(f"Camera {i} initialized successfully: {w}x{h}")
                return True
            else:
                camera.release()
        
    print("Error: Could not initialize any camera")
    return False

def generate_frames():
    global camera
    while True:
        with camera_lock:
            success, frame = camera.read()
        
        if not success or frame is None:
            print("Failed to capture frame")
            time.sleep(0.1)
            continue
            
        try:
            # Resize frame if too large
            h, w = frame.shape[:2]
            if w > 1280 or h > 720:
                frame = cv2.resize(frame, (640, 480))
                print(f"Resized frame from {w}x{h} to 640x480")
            
            # Convert colorspace if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Encode frame as JPEG with error handling
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            
            if not ret:
                print("Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            
            # Yield frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            print(f"Error processing frame: {e}")
            time.sleep(0.1)
            continue

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    global camera
    if camera and camera.isOpened():
        return "Camera is running"
    else:
        return "Camera is not available"

@app.route('/')
def index():
    return '''
    <html>
    <body>
    <h1>RPi Camera Stream</h1>
    <img src="/video_feed" width="640" height="480">
    <br><br>
    <a href="/status">Check Status</a>
    </body>
    </html>
    '''

if __name__ == '__main__':
    if initialize_camera():
        print("Starting Flask server...")
        try:
            app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            if camera:
                camera.release()
    else:
        print("Failed to initialize camera")
