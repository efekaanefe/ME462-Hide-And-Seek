# -*- encoding: UTF-8 -*-
from naoqi import ALProxy

nao_ip = "192.168.0.202"
port = 9559

life = ALProxy("ALAutonomousLife", nao_ip, port)
current = life.getState()
print(current)
life.setState("disabled")

