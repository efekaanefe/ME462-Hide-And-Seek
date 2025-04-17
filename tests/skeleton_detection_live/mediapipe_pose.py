import cv2
import mediapipe as mp
import time
import numpy as np

def main():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Configure MediaPipe Pose
    pose = mp_pose.Pose(
        static_image_mode=False,        # Set to False for video processing
        model_complexity=1,             # 0=Lite, 1=Full, 2=Heavy
        smooth_landmarks=True,          # Reduces jitter
        enable_segmentation=False,      # Set to True if you want segmentation
        min_detection_confidence=0.5,   # Minimum confidence for detection
        min_tracking_confidence=0.5     # Minimum confidence for tracking
    )
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set webcam resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # FPS calculation variables
    prev_time = 0
    curr_time = 0
    fps = 0
    fps_history = []
    
    print("Starting MediaPipe pose estimation. Press 'q' to quit.")
    
    try:
        while cap.isOpened():
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Start time for FPS calculation
            curr_time = time.time()
            
            # Convert the BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe Pose
            results = pose.process(frame_rgb)
            
            # Calculate FPS
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            # Maintain a rolling average of FPS
            fps_history.append(fps)
            if len(fps_history) > 30:  # Average over 30 frames
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            
            # Draw pose landmarks on the image
            if results.pose_landmarks:
                # Create a copy to draw on
                annotated_image = frame.copy()
                
                # Draw the pose landmarks
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Add FPS text to image
                cv2.putText(annotated_image, f"FPS: {avg_fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show the output image
                cv2.imshow("MediaPipe Live Pose Estimation", annotated_image)
            else:
                # If no pose detected, show original frame with FPS
                cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "No pose detected", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("MediaPipe Live Pose Estimation", frame)
            
            # Print FPS to console
            print(f"Current FPS: {fps:.2f} | Average FPS: {avg_fps:.2f}", end="\r")
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"Error during processing: {e}")
    
    finally:
        # Clean up
        pose.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\nPose estimation stopped.")

if __name__ == "__main__":
    main()
