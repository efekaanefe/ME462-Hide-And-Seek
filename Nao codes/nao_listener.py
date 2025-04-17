# -*- encoding: UTF-8 -*-
import socket
import math
from naoqi import ALProxy

# NAO ayarları
PORT = 5005
IP = "0.0.0.0"  # Tüm IP’lerden bağlantı kabul et

# NAO motor kontrolü
motion = ALProxy("ALMotion", "127.0.0.1", 9559)
motion.setStiffnesses("Body", 1.0)

def deg2rad(deg):
    return deg * math.pi / 180.0

# TCP server başlat
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((IP, PORT))
server_socket.listen(1)
print("NAO server hazır, bağlantı bekleniyor...")

conn, addr = server_socket.accept()
print("Bağlandı:", addr)

try:
    while True:
        data = conn.recv(1024)
        if not data:
            break

        try:
            # Gelen veriyi çöz
            msg = data.decode('utf-8') if isinstance(data, bytes) else data
            parts = msg.strip().split(",")
            r_val = float(parts[0].split(":")[1])
            l_val = float(parts[1].split(":")[1])

            # Derece → Radyan
            r_rad = deg2rad(r_val)
            l_rad = deg2rad(l_val)

            # NAO komutu (ani hareket için, speed 0.2 gibi)
            motion.setAngles("RShoulderRoll", -r_rad, 0.2)  # Sağ için -rad
            motion.setAngles("LShoulderRoll", l_rad, 0.2)

            print("Uygulandı  →  R: %.1f° (%.2f rad),  L: %.1f° (%.2f rad)" %
                  (r_val, r_rad, l_val, l_rad))

        except Exception as e:
            print("Veri çözüm hatası:", e)

except KeyboardInterrupt:
    print("Durduruldu")

# Temizlik
conn.close()
server_socket.close()
