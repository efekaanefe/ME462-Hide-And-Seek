#!/usr/bin/env python3
import cv2
import time
import numpy as np
import json  # Add this import
from utils.tcp_client import TCPClient
from utils.homography_projector import HomographyProjector
from utils.orientation_detector import OrientationDetector
from utils.mqtt_publisher import MQTTPublisher
import configparser

# Optional: Map ArUco IDs to names
ARUCO_ID_NAMES = {
    7: "NAO",
    79:"Target"
}


def get_camera_ip(room: str, camera: str, config_path="ip_config.ini") -> str:
    config = configparser.ConfigParser()
    config.read(config_path)
    try:
        return config[room][camera]
    except KeyError:
        raise ValueError(f"No IP found for {room}.{camera}")


def run_aruco_tracking(host: str, port: int = 8080, room_index: int = 0, cam_index: int = 0):
    # Initialize TCP stream
    tcp_client = TCPClient(host, port)
    if not tcp_client.connect():
        print("Failed to connect to TCP server.")
        return

    # Initialize projector (2D map overlay)
    projector = HomographyProjector()
    projector.select_room(room_index)
    projector.select_camera(cam_index)

    publisher = MQTTPublisher(broker_address="test.mosquitto.org", room_index=room_index, camera_index=cam_index)
    publisher.connect()

    # ArUco dictionary and detector setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector_params = cv2.aruco.DetectorParameters()

    print("Aruco marker tracking started. Press 'q' to exit.")
    while tcp_client.is_connected():
        frame = tcp_client.get_frame()
        if frame is None:
            print("Warning: No frame received.")
            time.sleep(0.1)
            continue

        frame = cv2.resize(frame, (1920, 1080))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)

        detections = []
        if ids is not None:
            for i, marker_corners in enumerate(corners):
                c = marker_corners[0]
                center_x = int(np.mean(c[:, 0]))
                center_y = int(np.mean(c[:, 1]))

                x_min = int(np.min(c[:, 0]))
                y_min = int(np.min(c[:, 1]))
                x_max = int(np.max(c[:, 0]))
                y_max = int(np.max(c[:, 1]))

                # Orientation angle
                dx = c[1][0] - c[0][0]
                dy = c[1][1] - c[0][1]
                orientation_angle = np.arctan2(dy, dx)

                marker_id = int(ids[i])
                name = ARUCO_ID_NAMES.get(marker_id, f"ID {marker_id}")

                detections.append({
                    "id": marker_id,
                    "name": name,
                    "position": (center_x, center_y),
                    "corners": c.tolist(),
                    "bbox": [x_min, y_min, x_max, y_max],
                    "orientation": orientation_angle
                })

            # Map projection
            detections = projector.update(detections)

            print(detections)

            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Draw results
            for det in detections:
                cx, cy = det["position"]
                name = det.get("name", f"ID {det['id']}")
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

                if "orientation" in det:
                    angle_deg = np.degrees(det["orientation"])
                    cv2.putText(frame, f"{name} - {angle_deg:.1f}°", (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Orientation arrow
                    dx = int(50 * np.cos(det["orientation"]))
                    dy = int(50 * np.sin(det["orientation"]))
                    cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy), (255, 0, 0), 2)

                if "map_position" in det:
                    mx, my = det["map_position"]
                    print(f"{name} @ map: ({mx:.2f}, {my:.2f})")

            # Overlay 2D map in corner
            frame = projector.visualize(frame, detections)
            print("asdgasdgasdg")

            # MQTT publishing for each detection
            if publisher.is_connected:
                for det in detections:
                    cx, cy = det["position"]
                    detection_data = {
                        "track_id":det["id"],
                        "name": det["name"],
                        "x": float(det["map_position"][0]),
                        "y": float(det["map_position"][1]),
                        "orientation": float(det["map_orientation"]),
                        "timestamp": time.time()
                    }

                    # Publish with marker ID as part of the topic
                    topic = f"tracking/{room_index}/{cam_index}/{detection_data['track_id']}"
                    publisher.publish(topic, json.dumps(detection_data), qos=1)
                    
                    print(f"Published: {det['name']} at ({cx}, {cy}) with orientation {np.degrees(det['orientation']):.1f}°")

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Aruco Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tcp_client.disconnect()
    cv2.destroyAllWindows()
    print("Tracking ended.")


def main():
    room = "room0"
    camera = "cam2"
    ip = get_camera_ip(room, camera)

    run_aruco_tracking(
        host=ip,
        port=8080,
        room_index=0,
        cam_index=2
    )


if __name__ == "__main__":
    main()