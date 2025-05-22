import cv2
import numpy as np
import time

from orientation_test_2D import PersonOrientationDetector
from homography_modular import HomographyTool

from tracker import PersonTracker

from MQTTPublisher import MQTTPublisher

def main():
    # Initialize the tools
    homography_tool = HomographyTool()
    orientation_detector = PersonOrientationDetector()
    person_tracker = PersonTracker()

    # MQTT
    publisher = MQTTPublisher(broker_address="mqtt.eclipseprojects.io")
    publisher.connect()
    
    # Configuration
    room_index = 0  # Room index to use
    cam_index = 0   # Camera index to use
    video_path = "test-home2.mp4"  # Path to your video file
    output_path = "output_tracking.mp4"  # Path to save the output video

    # Load 2D map
    map_path = f"rooms_database/room{room_index}/2Dmap.png"
    map_img = cv2.imread(map_path)
    map_img = cv2.resize(map_img, (300, 300))

    # Load homography matrices
    homography_tool.load_homography_matrices("homography_matrices.json")

    # Select room and camera
    homography_tool.select_room(room_index)
    homography_tool.select_camera(cam_index)

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    out = cv2.VideoWriter(output_path, fourcc, fps, (1600, 1200))

    # Initialize visualization window
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking", 1280, 720)

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate frame 90 degrees counterclockwise
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Resize frame to 1600x1200
        frame = cv2.resize(frame, (1600, 1200))

        # Convert frame to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect people and their orientations
        people = orientation_detector.detect_people(frame_rgb)

        # Convert detections to format expected by tracker
        detections = []
        for person in people:
            x, y, w, h = person["bbox"]
            detections.append({
                "bbox": [x, y, x + w, y + h],
                "confidence": person.get("confidence", 0.5)
            })

        # Update tracker
        current_time = time.time()
        tracks = person_tracker.update(frame, detections, current_time)

        # Map tracked people to 2D coordinates
        mapped_people = []
        for track_id, track_data in tracks.items():
            if track_data.get('active', False):
                bbox = track_data['bbox']
                # Convert bbox to center point
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = bbox[3]  # Use bottom center as foot position

                # Find matching person from orientation detector
                matching_person = None
                for person in people:
                    if (abs(person["bbox"][0] - bbox[0]) < 10 and 
                        abs(person["bbox"][1] - bbox[1]) < 10):
                        matching_person = person
                        break

                # Map to 2D coordinates
                try:
                    mapped_point = cv2.perspectiveTransform(
                        np.array([[[center_x, center_y]]], dtype=np.float32),
                        np.array(homography_tool.homography_matrices[f"room{room_index}_cam{cam_index}"]["matrix"])
                    )[0][0]

                    # Get orientation if available
                    orientation = None
                    
                    # Calculate movement-based orientation from tracking history
                    movement_orientation = None
                    if 'history' in track_data and len(track_data['history']) >= 2:
                        # Get last two positions from history
                        prev_pos = track_data['history'][-2]
                        curr_pos = track_data['history'][-1]
                        
                        # Calculate movement vector
                        dx = curr_pos[0] - prev_pos[0]
                        dy = curr_pos[1] - prev_pos[1]
                        
                        # Only use movement if it's significant enough
                        movement_magnitude = np.sqrt(dx*dx + dy*dy)
                        if movement_magnitude > 1:  # Minimum movement threshold
                            movement_orientation = np.arctan2(dy, dx)

                    # Combine MediaPipe orientation with movement-based orientation
                    if matching_person and "orientation" in matching_person:
                        # Transform MediaPipe orientation to map coordinates
                        dx, dy = np.cos(matching_person["orientation"]), np.sin(matching_person["orientation"])
                        cam_dir_point = (center_x + dx * 20, center_y + dy * 20)
                        
                        # Transform direction point to map coordinates
                        map_dir_point = cv2.perspectiveTransform(
                            np.array([[[cam_dir_point[0], cam_dir_point[1]]]], dtype=np.float32),
                            np.array(homography_tool.homography_matrices[f"room{room_index}_cam{cam_index}"]["matrix"])
                        )[0][0]
                        
                        # Calculate orientation in map coordinates
                        map_dx, map_dy = map_dir_point[0] - mapped_point[0], map_dir_point[1] - mapped_point[1]
                        mediapipe_orientation = np.arctan2(map_dy, map_dx)

                        # Combine orientations if both are available
                        if movement_orientation is not None:
                            # Weight the orientations (adjust weights as needed)
                            mediapipe_weight = 0.0
                            movement_weight = 1.0
                            
                            # Calculate weighted average of orientations
                            # Use complex numbers for proper angle averaging
                            mediapipe_complex = np.exp(1j * mediapipe_orientation)
                            movement_complex = np.exp(1j * movement_orientation)
                            
                            combined_complex = (mediapipe_weight * mediapipe_complex + 
                                              movement_weight * movement_complex)
                            orientation = np.angle(combined_complex)
                        else:
                            orientation = mediapipe_orientation
                    elif movement_orientation is not None:
                        # Use movement-based orientation if MediaPipe orientation is not available
                        orientation = movement_orientation

                    mapped_people.append({
                        "track_id": track_id,
                        "name": track_data.get('name', 'Unknown'),
                        "position": (mapped_point[0], mapped_point[1]),
                        "bbox": bbox,
                        "orientation": orientation,
                        "mediapipe_orientation": matching_person["orientation"] if matching_person and "orientation" in matching_person else None,
                        "movement_orientation": movement_orientation
                    })
                except Exception as e:
                    print(f"Error mapping point: {e}")

        # Draw tracking results on frame
        for person in mapped_people:
            track_id = person["track_id"]
            name = person["name"]
            bbox = person["bbox"]
            map_pos = person["position"]
            orientation = person.get("orientation")
            mediapipe_orientation = person.get("mediapipe_orientation")
            movement_orientation = person.get("movement_orientation")

            # Draw bounding box
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

            # Draw label
            label = f"{name} (ID: {track_id})"
            cv2.putText(frame, label, (int(bbox[0]), int(bbox[1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw map coordinates
            coord_text = f"Map: ({int(map_pos[0])}, {int(map_pos[1])})"
            cv2.putText(frame, coord_text, (int(bbox[0]), int(bbox[3] + 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw orientation if available
            if orientation is not None:
                # Draw combined orientation arrow on frame (red)
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = bbox[3]
                arrow_length = 30
                dx = int(arrow_length * np.cos(orientation))
                dy = int(arrow_length * np.sin(orientation))
                cv2.arrowedLine(frame, (int(center_x), int(center_y)),
                              (int(center_x + dx), int(center_y + dy)),
                              (0, 0, 255), 2)

                # Draw MediaPipe orientation if available (blue)
                if mediapipe_orientation is not None:
                    dx = int(arrow_length * np.cos(mediapipe_orientation))
                    dy = int(arrow_length * np.sin(mediapipe_orientation))
                    cv2.arrowedLine(frame, (int(center_x), int(center_y)),
                                  (int(center_x + dx), int(center_y + dy)),
                                  (255, 0, 0), 1)

                # Draw movement orientation if available (green)
                if movement_orientation is not None:
                    dx = int(arrow_length * np.cos(movement_orientation))
                    dy = int(arrow_length * np.sin(movement_orientation))
                    cv2.arrowedLine(frame, (int(center_x), int(center_y)),
                                  (int(center_x + dx), int(center_y + dy)),
                                  (0, 255, 0), 1)

            # Draw position and orientation on map
            map_copy = map_img.copy()
            # Scale map coordinates to map image size
            map_x = int(map_pos[0] * map_img.shape[1] / 1000)  # Assuming map coordinates are in 0-1000 range
            map_y = int(map_pos[1] * map_img.shape[0] / 1000)
            
            # Draw person circle
            cv2.circle(map_copy, (map_x, map_y), 5, (0, 0, 255), -1)
            cv2.putText(map_copy, f"ID: {track_id}", (map_x + 5, map_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Draw orientation arrow on map if available
            if orientation is not None:
                arrow_length = 15
                dx = int(arrow_length * np.cos(orientation))
                dy = int(arrow_length * np.sin(orientation))
                cv2.arrowedLine(map_copy, (map_x, map_y),
                              (map_x + dx, map_y + dy),
                              (0, 255, 0), 2)

            # Publish coordinates and orientation to MQTT
            position_data = {
                "track_id": track_id,
                "name": name,
                "x": float(map_pos[0]),
                "y": float(map_pos[1]),
                "orientation": float(orientation) if orientation is not None else None,
                "timestamp": current_time
            }
            publisher.publish(f"game/player/position/{track_id}", str(position_data), qos=1)

        # Overlay map on frame
        map_overlay = np.zeros_like(frame)
        map_overlay[10:10+map_img.shape[0], 10:10+map_img.shape[1]] = map_copy
        frame = cv2.addWeighted(frame, 1, map_overlay, 0.7, 0)

        # Write frame to output video
        out.write(frame)

        # Display frame
        cv2.imshow("Tracking", frame)

        # Print FPS every 30 frames
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Print tracking statistics
    time_data = person_tracker.get_time_data()
    print("\nTracking Statistics:")
    for data in time_data:
        print(data)

if __name__ == "__main__":
    main() 