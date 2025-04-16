# People Tracking GUI

## How it works?
1.	Detector (YOLOV8) finds objects → adds to detections
2.	Tracker (Tracker) matches them with existing tracks
3.	If unmatched, optional face recognition
4.	Tracks updated → tracked_objects
5.	Display thread grabs processed frame
6.	Displays on GUI with FPS and controls


## The tracking system works with these major components:
Main GUI Thread:
    Handles the start tracking button click
    Validates prerequisites (cameras, YOLO model)
    Starts cameras and launches worker threads
    Updates UI elements
Processing Thread:
    Gets frames from each camera's queue
    Detects people using YOLO model
    Updates the PersonTracker with new detections
    Tracks Re-ID statistics before and after updates
    Logs significant events (identity switches, new IDs)
    Draws bounding boxes and information on frames
    Places processed frames in output queues
Display Thread:
    Retrieves processed frames from output queues
    Prepares images for display (resizing, format conversion)
    Schedules UI updates on the main thread
    Controls display framerate
PersonTracker Update Process:
    Matches detections with existing tracks using IoU and appearance
    Periodically performs face detection for better identification
    Re-identifies people based on face features and similarity scores
    Handles identity switches and confirmations
    Manages track creation, updating, and inactivation
    Maintains statistics about identifications and switches
