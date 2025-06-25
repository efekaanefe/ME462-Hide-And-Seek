#!/usr/bin/env python3
import socket
import cv2
import numpy as np
import struct

class VideoStreamClient:
    def __init__(self, host, port=8080):
        self.host = host
        self.port = port

    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        print(f"[CLIENT] Connected to {self.host}:{self.port}")

        try:
            while True:
                # Receive size first
                size_data = self._recv_exact(sock, 4)
                if not size_data:
                    break
                size = struct.unpack('!I', size_data)[0]

                # Receive frame
                data = self._recv_exact(sock, size)
                if not data:
                    break

                # Decode JPEG
                frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

                # Show frame
                cv2.imshow("Video Stream", frame)
                if cv2.waitKey(1) == ord('q'):
                    break

        finally:
            sock.close()
            cv2.destroyAllWindows()

    def _recv_exact(self, sock, size):
        """Receive exact bytes"""
        buf = b''
        while len(buf) < size:
            part = sock.recv(size - len(buf))
            if not part:
                return None
            buf += part
        return buf

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('host', help='Server IP (Raspberry Pi)')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    client = VideoStreamClient(args.host, args.port)
    client.start()
