# -*- coding: utf-8 -*-

import socket
import json
from naoqi import ALProxy

NAO_IP = "127.0.0.1"
PORT = 12345

motion = ALProxy("ALMotion", NAO_IP, 9559)
posture = ALProxy("ALRobotPosture", NAO_IP, 9559)

server = socket.socket()
server.bind(("0.0.0.0", PORT))
server.listen(1)
print("Komut bekleniyor...")

while True:
    client, addr = server.accept()
    print("Bağlantı kuruldu:", addr)

    data = client.recv(1024).decode()
    if not data:
        client.close()
        continue

    try:
        msg = json.loads(data)

        if msg["type"] == "move":
            x = msg["x"]
            y = msg["y"]
            theta = msg["theta"]

            print("Yürüyüş pozisyonuna geçiliyor...")
            posture.goToPosture("StandInit", 0.5)

            print("Yürüyüş başlıyor:", x, y, theta)
            motion.moveTo(x, y, theta)

    except Exception as e:
        print("Hata:", e)

    client.close()
