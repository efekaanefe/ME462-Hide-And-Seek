from flask import Flask, Response, render_template
import cv2

app = Flask(_name_)
camera = cv2.VideoCapture(0)  # 0 = default webcam
print("Width:", camera.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Çözünürlük ayarla (örnek: 1280x720)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Görüntüyü JPEG formatına çevir
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # MJPEG formatında ver
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if _name_ == "_main_":
    app.run(debug=True)