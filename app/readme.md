# People Tracking GUI

## How it works?
1.	Detector (YOLOV8) finds objects → adds to detections
2.	Tracker (Tracker) matches them with existing tracks
3.	If unmatched, optional face recognition
4.	Tracks updated → tracked_objects
5.	Display thread grabs processed frame
6.	Displays on GUI with FPS and controls


# Multi-Threaded Tracking System Overview
## Main GUI Thread
- Handles the **Start Tracking** button click
- Validates prerequisites:
  - Cameras
  - YOLO model
- Starts cameras and launches worker threads
- Updates UI elements

## Processing Thread
- Gets frames from each camera's queue
- Detects people using the YOLO model
- Updates the `PersonTracker` with new detections
- Tracks Re-ID statistics before and after updates
- Logs significant events:
  - Identity switches
  - New IDs
- Draws bounding boxes and information on frames
- Places processed frames in output queues

## Display Thread
- Retrieves processed frames from output queues
- Prepares images for display:
  - Resizing
  - Format conversion
- Schedules UI updates on the main thread
- Controls display framerate

## PersonTracker Update Process
- Matches detections with existing tracks using:
  - IoU (Intersection over Union)
  - Appearance features
- Periodically performs face detection for better identification
- Re-identifies people based on:
  - Face features
  - Similarity scores
- Handles identity switches and confirmations
- Manages:
  - Track creation
  - Track updating
  - Track inactivation
- Maintains statistics about:
  - Identifications
  - Identity switches
