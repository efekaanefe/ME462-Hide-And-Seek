# -*- coding: utf-8 -*-

# PC (Server): NAO'dan görüntü alır ve gösterir
import socket
import cv2
import numpy as np

s = socket.socket()
s.bind(("", 8000))  # Her IP'den bağlantı al
s.listen(1)
conn, addr = s.accept()
print("Baglanti geldi:", addr)

while True:
    # 4 byte uzunluk bilgisi
    length_data = conn.recv(4)
    if not length_data:
        break

    size = int.from_bytes(length_data, byteorder='big')

    data = b""
    while len(data) < size:
        packet = conn.recv(size - len(data))
        if not packet:
            break
        data += packet

    img_array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is not None:
        cv2.imshow("NAO Camera", frame)

    if cv2.waitKey(1) == 27:  # ESC ile çık
        break

conn.close()
cv2.destroyAllWindows()
