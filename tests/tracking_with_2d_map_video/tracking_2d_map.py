import cv2
import numpy as np
import time

from orientation_test_2D import PersonOrientationDetector
from homography_modular import HomographyTool
from tracker import PersonTracker
from MQTTPublisher import MQTTPublisher

def run_tracking(video_path: str, output_path: str, room_index: int = 0, cam_index: int = 0, 
                headless: bool = False, show_fps: bool = True, use_depth_orientation: bool = True):
    """
    Run the tracking system with specified parameters.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save output video
        room_index: Room index to use
        cam_index: Camera index to use
        headless: If True, run without displaying video window
        show_fps: If True, display FPS on video
    """
    # Initialize the tools
    homography_tool = HomographyTool()
    orientation_detector = PersonOrientationDetector(use_depth_orientation=use_depth_orientation)
    person_tracker = PersonTracker()

    # MQTT
    publisher = MQTTPublisher(broker_address="mqtt.eclipseprojects.io")
    publisher.connect()
    
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
    out = cv2.VideoWriter(output_path, fourcc, fps, (1600, 1200))

    # Initialize visualization window if not headless
    if not headless:
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking", 1280, 720)

    # Initialize FPS calculation variables
    fps = 0
    fps_list = []  # Store last 30 FPS values for smoothing

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Start timing for FPS calculation
        frame_start_time = time.time()

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

        # Calculate FPS
        frame_time = time.time() - frame_start_time
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        
        # Update FPS list for smoothing
        fps_list.append(current_fps)
        if len(fps_list) > 30:  # Keep last 30 frames
            fps_list.pop(0)
        fps = sum(fps_list) / len(fps_list)  # Calculate average FPS

        # Draw FPS on frame if enabled
        if show_fps:
            # Get frame dimensions
            height, width = frame.shape[:2]
            # Calculate text size to position it properly
            text = f"FPS: {fps:.1f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            # Position text in top right corner with padding
            x = width - text_width - 20  # 20 pixels padding from right edge
            y = text_height + 20  # 20 pixels padding from top
            cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 0), thickness)

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
                    axis_info = None
                    
                    # Use only MediaPipe orientation
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
                        orientation = np.arctan2(map_dy, map_dx)
                        
                        # Get axis info if available
                        if "axis_info" in matching_person:
                            axis_info = matching_person["axis_info"]

                    mapped_people.append({
                        "track_id": track_id,
                        "name": track_data.get('name', 'Unknown'),
                        "position": (mapped_point[0], mapped_point[1]),
                        "bbox": bbox,
                        "orientation": orientation,
                        "axis_info": axis_info
                    })
                except Exception as e:
                    print(f"Error mapping point: {e}")

        # Draw tracking results on frame if not headless
        for person in mapped_people:
            track_id = person["track_id"]
            name = person["name"]
            bbox = person["bbox"]
            map_pos = person["position"]
            orientation = person.get("orientation")

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
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

            # Draw orientation if available
            if orientation is not None and "axis_info" in person:
                axis_info = person["axis_info"]
                center = axis_info["center"]
                
                # Draw x-axis (red) - forward direction
                cv2.arrowedLine(frame, center, axis_info["x_axis"], (0, 0, 255), 2)
                
                # Draw y-axis (green) - perpendicular to x-axis
                cv2.arrowedLine(frame, center, axis_info["y_axis"], (0, 255, 0), 2)
                
                # Draw z-axis (blue) - opposite to x-axis
                cv2.arrowedLine(frame, center, axis_info["z_axis"], (255, 0, 0), 2)

            # Draw position and orientation on map
            map_copy = map_img.copy()
            # Scale map coordinates to map image size
            map_x = int(map_pos[0] * map_img.shape[1] / 1000)  # Assuming map coordinates are in 0-1000 range
            map_y = int(map_pos[1] * map_img.shape[0] / 1000)
            
            # Draw person circle
            cv2.circle(map_copy, (map_x, map_y), 5, (0, 0, 0), -1)
            cv2.putText(map_copy, f"ID: {track_id}", (map_x + 5, map_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
            
            # Draw orientation arrow on map if available
            if orientation is not None:
                arrow_length = 15
                dx = int(arrow_length * np.cos(orientation))
                dy = int(arrow_length * np.sin(orientation))
                cv2.arrowedLine(map_copy, (map_x, map_y),
                                (map_x + dx, map_y + dy),
                                (0, 0, 0), 4)

            # Overlay map on frame
            map_overlay = np.zeros_like(frame)
            map_overlay[10:10+map_img.shape[0], 10:10+map_img.shape[1]] = map_copy
            frame = cv2.addWeighted(frame, 1, map_overlay, 1, 0)

            # Display orientation angle on frame if available
            if orientation is not None:
                orientation_deg = np.degrees(orientation)
                orientation_text = f"Orientation: {orientation_deg:.1f}°"
                cv2.putText(frame, orientation_text, (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        # Publish coordinates and orientation to MQTT
        for person in mapped_people:
            position_data = {
                "track_id": person["track_id"],
                "name": person["name"],
                "x": float(person["position"][0]),
                "y": float(person["position"][1]),
                "orientation": float(person["orientation"]) if person["orientation"] is not None else None,
                "timestamp": current_time
            }
            publisher.publish(f"game/player/position/{person['track_id']}", str(position_data), qos=1)

        # Write frame to output video
        out.write(frame)

        # Display frame if not headless
        if not headless:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    cap.release()
    out.release()
    if not headless:
        cv2.destroyAllWindows()

    # Print tracking statistics
    time_data = person_tracker.get_time_data()
    print("\nTracking Statistics:")
    for data in time_data:
        print(data)

def main():
    # Example usage with direct function call
    run_tracking(
        video_path="test-home2.mp4",
        output_path="output_tracking.mp4",
        room_index=0,
        cam_index=0,
        headless=True,  # Set to True for headless mode
        show_fps=True,   # Set to False to hide FPS
        use_depth_orientation=False
    )

if __name__ == "__main__":
    main() 