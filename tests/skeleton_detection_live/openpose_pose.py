# TODO: I couldn't install openpose since build without gpu not implemented yet, I guess...

import sys
import time
import cv2
import numpy as np
from openpose import pyopenpose as op

def main():
    # Configure OpenPose parameters
    params = dict()
    params["model_folder"] = "models/"  # Path to the OpenPose models
    params["net_resolution"] = "-1x368"  # Use "-1x368" for higher accuracy, "-1x256" for faster performance
    
    # Initialize OpenPose
    try:
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        
        # Create OpenPose datum object
        datum = op.Datum()
    except Exception as e:
        print(f"Error initializing OpenPose: {e}")
        return

    # Start capturing video
    cap = cv2.VideoCapture(0)  # Use 0 for webcam
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # FPS calculation variables
    prev_time = 0
    curr_time = 0
    fps = 0
    fps_history = []
    
    print("Starting pose estimation. Press 'q' to quit.")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Start time for FPS calculation
            curr_time = time.time()
            
            # Process frame with OpenPose
            datum.cvInputData = frame
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            
            # Calculate FPS
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            # Maintain a rolling average of FPS
            fps_history.append(fps)
            if len(fps_history) > 30:  # Average over 30 frames
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            
            # Display results
            output_image = datum.cvOutputData
            
            # Add FPS text to image
            cv2.putText(output_image, f"FPS: {avg_fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show the output image
            cv2.imshow("OpenPose Live Pose Estimation", output_image)
            
            # Print FPS to console
            print(f"Current FPS: {fps:.2f} | Average FPS: {avg_fps:.2f}", end="\r")
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"Error during processing: {e}")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("\nPose estimation stopped.")

if __name__ == "__main__":
    main()
