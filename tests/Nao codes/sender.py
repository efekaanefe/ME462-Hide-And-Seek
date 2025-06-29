# -*- coding: utf-8 -*-

import socket
import json
import math
NAO_IP = "192.168.0.202"
PORT = 12345

client = socket.socket()
client.connect((NAO_IP, PORT))

command = {
    "type": "move",
    "x": 0.0,
    "y": 0.0,
    "theta": 0.0
}

client.send(json.dumps(command).encode())
client.close()
