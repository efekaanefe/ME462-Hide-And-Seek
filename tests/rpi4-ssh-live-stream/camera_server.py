import cv2
from flask import Flask, Response
import threading
import time

app = Flask(__name__)
camera = None
camera_lock = threading.Lock()

def initialize_camera():
    global camera
    # Try different camera indices if needed (0, 1, 2...)
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    if not camera.isOpened():
        print("Error: Could not open camera")
        return False
    return True

def generate_frames():
    global camera
    while True:
        with camera_lock:
            success, frame = camera.read()
        
        if not success:
            break
        else:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame = buffer.tobytes()
            
            # Yield frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
    <body>
    <h1>RPi Camera Stream</h1>
    <img src="/video_feed" width="640" height="480">
    </body>
    </html>
    '''

if __name__ == '__main__':
    if initialize_camera():
        app.run(host='0.0.0.0', port=5000, threaded=True)
    else:
        print("Failed to initialize camera")
