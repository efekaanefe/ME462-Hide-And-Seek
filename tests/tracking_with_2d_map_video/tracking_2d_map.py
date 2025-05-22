import cv2
import numpy as np
import time
import sys
from pathlib import Path

from orientation_test_2D import PersonOrientationDetector
from homography_modular import HomographyTool

from people_tracking import PersonTracker


def main():
    # Initialize the tools
    homography_tool = HomographyTool()
    orientation_detector = PersonOrientationDetector()
    person_tracker = PersonTracker()

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

                # Map to 2D coordinates
                try:
                    mapped_point = cv2.perspectiveTransform(
                        np.array([[[center_x, center_y]]], dtype=np.float32),
                        np.array(homography_tool.homography_matrices[f"room{room_index}_cam{cam_index}"]["matrix"])
                    )[0][0]

                    mapped_people.append({
                        "track_id": track_id,
                        "name": track_data.get('name', 'Unknown'),
                        "position": (mapped_point[0], mapped_point[1]),
                        "bbox": bbox
                    })
                except Exception as e:
                    print(f"Error mapping point: {e}")

        # Draw tracking results on frame
        for person in mapped_people:
            track_id = person["track_id"]
            name = person["name"]
            bbox = person["bbox"]
            map_pos = person["position"]

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

            # Draw position on map
            map_copy = map_img.copy()
            # Scale map coordinates to map image size
            map_x = int(map_pos[0] * map_img.shape[1] / 1000)  # Assuming map coordinates are in 0-1000 range
            map_y = int(map_pos[1] * map_img.shape[0] / 1000)
            cv2.circle(map_copy, (map_x, map_y), 5, (0, 0, 255), -1)
            cv2.putText(map_copy, f"ID: {track_id}", (map_x + 5, map_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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