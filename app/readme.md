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



## The Tracking Process Step by Step
Camera Capture: Each camera runs in its own thread capturing frames
Object Detection: YOLO detects people in frames
Track Matching: Uses Hungarian algorithm to match detections with existing tracks
Face Detection & Recognition:
Only runs periodically (every N frames) to save processing power
Extracts face features using ArcFace
Compares with known database and existing tracks
Re-identification:
Periodically checks if a tracked person still matches their assigned identity
Has a failure threshold to handle potential identity switches
Visual Results: Draw bounding boxes and identity information on the frame

