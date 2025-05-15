# -*- encoding: UTF-8 -*-
from naoqi import ALProxy
import socket
import math

# NAO ayarları
NAO_IP = "127.0.0.1"
PORT = 9559

# Hareket servisi başlat
motion = ALProxy("ALMotion", NAO_IP, PORT)
motion.setStiffnesses("Head", 1.0)

# Socket başlat
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", 9999))  # NAO bu portta PC'yi bekler
server.listen(1)
print("Head control server ready. Waiting for connection on port 9999...")

conn, addr = server.accept()
print("Connection accepted from {}".format(addr))

try:
    while True:
        data = conn.recv(1024)
        if not data:
            break

        # Gelen veriyi temizle
        message = data.decode('utf-8').strip()
        message = message.replace("(", "").replace(")", "").replace("\n", "")

        try:
            yaw_deg, pitch_deg = map(float, message.split(","))

            # Dereceyi radyana çevir
            yaw_rad = math.radians(yaw_deg)
            pitch_rad = math.radians(pitch_deg)

            # Robotun kafasını döndür
            motion.setAngles(["HeadYaw", "HeadPitch"], [yaw_rad, pitch_rad], 0.2)

            print("Baş hareketi: Yaw = {:.2f}°, Pitch = {:.2f}°".format(yaw_deg, pitch_deg))

        except Exception as e:
            print("Veri işlenemedi:", e)

except KeyboardInterrupt:
    print("Kapatılıyor...")

finally:
    conn.close()
    server.close()
    motion.setStiffnesses("Head", 0.0)
    print("Bağlantı kapatıldı, stiffness sıfırlandı.")
