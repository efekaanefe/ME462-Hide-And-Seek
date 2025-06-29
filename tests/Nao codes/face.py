# -*- encoding: UTF-8 -*-

"""
This example shows how to use ALTracker with face.
"""

import time
import argparse
from naoqi import ALProxy

def main(IP, PORT, faceSize):

    print "Connecting to", IP, "with port", PORT
    motion = ALProxy("ALMotion", IP, PORT)
    tracker = ALProxy("ALTracker", IP, PORT)
    posture = ALProxy("ALRobotPosture", IP, PORT)

    # First, wake up.
    motion.setStiffnesses("Body", 1.0)
    time.sleep(1)
    posture.goToPosture("StandInit",0.5)
    time.sleep(1)
    motion.setStiffnesses("Head", 1.0)
    time.sleep(1)
    motion.setAngles("HeadPitch", -0.5, 0.5)
    # Add target to track.
    targetName = "Face"
    faceWidth = faceSize
    tracker.registerTarget(targetName, faceWidth)

    # Then, start tracker.
    tracker.track(targetName)
    tracker.setMode("Move")

    print "ALTracker successfully started, now show your face to robot!"
    print "Use Ctrl+c to stop this script."

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print
        print "Interrupted by user"
        print "Stopping..."

    # Stop tracker.
    tracker.stopTracker()
    motion.stopMove()
    tracker.unregisterAllTargets()
    motion.rest()

    print "ALTracker stopped."


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.0.202",
                        help="Robot ip address.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number.")
    parser.add_argument("--facesize", type=float, default=0.1,
                        help="Face width.")

    args = parser.parse_args()

    main(args.ip, args.port, args.facesize)