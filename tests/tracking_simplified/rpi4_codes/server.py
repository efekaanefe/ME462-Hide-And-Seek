#!/usr/bin/env python3
import cv2
import socket
import struct
import threading

class VideoStreamServer:
    def __init__(self, host='0.0.0.0', port=8080, width=1280, height=720, fps=30):
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_FPS, fps)
        self.running = False

    def start(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(1)
        print(f"[SERVER] Listening on {self.host}:{self.port}")
        self.running = True

        while self.running:
            client_socket, addr = server_socket.accept()
            print(f"[SERVER] Client connected: {addr}")
            client_thread = threading.Thread(target=self._handle_client, args=(client_socket,))
            client_thread.start()

    def _handle_client(self, client_socket):
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    print("[SERVER] Failed to capture frame")
                    continue

                # Optional: resize down to reduce bandwidth
                frame = cv2.resize(frame, (self.width, self.height))

                # Encode to JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # lower = smaller file
                result, encoded_image = cv2.imencode('.jpg', frame, encode_param)
                data = encoded_image.tobytes()

                # Send size then data
                size = struct.pack('!I', len(data))
                client_socket.sendall(size + data)
        except Exception as e:
            print(f"[SERVER] Client connection closed: {e}")
        finally:
            client_socket.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    server = VideoStreamServer(args.host, args.port, args.width, args.height, args.fps)
    server.start()
