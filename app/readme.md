# People Tracking GUI

## How it works?
1.	Detector (YOLOV8) finds objects → adds to detections
2.	Tracker (Tracker) matches them with existing tracks
3.	If unmatched, optional face recognition
4.	Tracks updated → tracked_objects
5.	Display thread grabs processed frame
6.	Displays on GUI with FPS and controls



