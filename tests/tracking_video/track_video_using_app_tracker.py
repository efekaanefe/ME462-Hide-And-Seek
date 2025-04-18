import sys
import os
import cv2
import numpy as np
import time
from pathlib import Path

# Add app directory to sys.path to import from app module
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from app.people_tracking import PersonTracker, model, face_model  # Import tracker and models from main app

class SafePersonTracker(PersonTracker):
    """Wrapper around PersonTracker to handle errors safely"""
    
    def __init__(self, reid_interval=5.0):
        """Initialize with custom re-identification interval"""
        super().__init__()
        # Set custom re-identification interval
        self.periodic_reid_interval = reid_interval
        self.reid_attempts = {}  # Track re-ID attempts: {track_id: last_attempt_time}
        print(f"Re-identification interval set to {reid_interval} seconds")
    
    def update(self, frame, detections, current_time=None):
        """Safe wrapper for update method to handle errors"""
        try:
            tracked_objects = super().update(frame, detections, current_time)
            
            # Record re-ID attempts for visualization
            for track_id, track_data in tracked_objects.items():
                if track_id in self.last_reid_checks:
                    self.reid_attempts[track_id] = self.last_reid_checks[track_id]
            
            return tracked_objects
        except ValueError as e:
            if "cost matrix is infeasible" in str(e):
                print("Warning: Caught 'cost matrix is infeasible' error, returning empty tracking results")
                # Return empty tracking results
                return {}
            else:
                # Re-raise if it's a different ValueError
                print(f"Warning: Caught unexpected ValueError: {e}, returning empty tracking results")
                return {}
        except Exception as e:
            # Catch any other exceptions during tracking
            print(f"Warning: Error during tracking: {e}, returning empty tracking results")
            return {}
    
    def _hungarian_match_iou_appearance(self, frame, tracks, detections, current_time):
        """Safe wrapper for Hungarian matching to handle errors"""
        try:
            return super()._hungarian_match_iou_appearance(frame, tracks, detections, current_time)
        except ValueError as e:
            if "cost matrix is infeasible" in str(e):
                print("Warning: Handling 'cost matrix is infeasible' error in Hungarian algorithm")
                # Return a safe default: no matches, all tracks and detections unmatched
                return [], list(tracks.keys()), list(range(len(detections)))
            else:
                # Re-raise if it's a different ValueError
                raise
    
    def get_next_reid_time(self, track_id, current_time):
        """Get time until next re-identification for a track"""
        if track_id not in self.last_reid_checks:
            return 0  # First check will happen soon
            
        last_check = self.last_reid_checks.get(track_id, 0)
        time_since_last = current_time - last_check
        time_until_next = max(0, self.periodic_reid_interval - time_since_last)
        return time_until_next

def process_video(video_path, output_path=None, skip_frames=0, confidence=0.5, headless=False, reid_interval=5.0):
    """
    Process a video file for people tracking
    
    Args:
        video_path: Path to input video file
        output_path: Path to save output video (if None, display only)
        skip_frames: Number of frames to skip between processing (for speed)
        confidence: Detection confidence threshold
        headless: Whether to run in headless mode (no GUI display)
        reid_interval: Interval in seconds between re-identification attempts
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps:.2f} fps, {total_frames} frames")
    
    # Set up output video if requested
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"Error: Could not create output video file {output_path}")
            out = None
        else:
            print(f"Writing output to {output_path}")
    else:
        out = None
    
    # Initialize tracker with error handling and custom reid interval
    tracker = SafePersonTracker(reid_interval)
    print(f"Loaded {len(tracker.known_people)} known people")
    
    # Process frames
    frame_idx = 0
    processing_times = []
    error_count = 0
    max_errors = 5  # Maximum number of consecutive errors before giving up
    
    try:
        while True:
            # Check if we've had too many consecutive errors
            if error_count > max_errors:
                print(f"Stopping due to {error_count} consecutive errors")
                break
                
            # Try to read next frame
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Reset error counter on successful frame read
                error_count = 0
            except Exception as e:
                print(f"Error reading frame: {e}")
                error_count += 1
                time.sleep(0.1)  # Short delay before retrying
                continue
            
            # Skip frames if requested (for faster processing)
            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue
            
            # Start timer
            start_time = time.time()
            
            # Detect people
            if model is None:
                print("Error: YOLO model not available. Cannot process video.")
                break
                
            # Run detection with error handling
            detections = []
            try:
                results = model(frame, conf=confidence, classes=[0], verbose=False)  # class 0 is person
                
                if results and len(results) > 0:
                    result = results[0]  # Get Boxes object
                    boxes = result.boxes.cpu().numpy()
                    
                    for box in boxes:
                        if int(box.cls[0]) == 0:  # Check class is person
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': conf
                            })
            except Exception as e:
                print(f"Error during detection: {e}")
                error_count += 1
                # Continue with empty detections rather than skipping frame
                detections = []
            
            # Track detections (SafePersonTracker handles errors internally)
            current_time = time.time()
            tracked_objects = tracker.update(frame, detections, current_time)
            
            # Draw results
            try:
                result_frame = draw_results(frame.copy(), tracked_objects, tracker, current_time)
            except Exception as e:
                print(f"Error drawing results: {e}")
                result_frame = frame.copy()  # Use original frame if drawing fails
                error_count += 1
            
            # Calculate FPS
            try:
                process_time = time.time() - start_time
                processing_times.append(process_time)
                
                # Add FPS info to frame (with extra error handling)
                avg_time = np.mean(processing_times[-20:]) if processing_times else 0
                fps_text = f"FPS: {1.0/avg_time:.1f}" if avg_time > 0 else "FPS: N/A"
                cv2.putText(result_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                
                # Add progress info
                progress = frame_idx / total_frames if total_frames > 0 else 0
                cv2.putText(result_frame, f"Progress: {progress:.1%}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add re-ID interval info
                cv2.putText(result_frame, f"Re-ID interval: {reid_interval}s", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error calculating stats: {e}")
                error_count += 1
            
            # Save frame to output video if requested
            if out:
                try:
                    out.write(result_frame)
                except Exception as e:
                    print(f"Error writing frame to output: {e}")
                    error_count += 1
            
            # Display frame if not in headless mode
            if not headless:
                try:
                    cv2.imshow('Tracking', result_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    print(f"Warning: Could not display frame: {e}")
                    # If display fails, switch to headless mode
                    headless = True
                    print("Switching to headless mode (no GUI display)")
            
            frame_idx += 1
            
            # Print progress occasionally
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames ({progress:.1%})")
    
    except KeyboardInterrupt:
        print("Processing interrupted by user")
    
    finally:
        # Clean up
        try:
            cap.release()
        except:
            pass
            
        if out:
            try:
                out.release()
            except:
                pass
                
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        # Print stats
        try:
            avg_time = np.mean(processing_times) if processing_times else 0
            print(f"Average processing time: {avg_time:.3f}s ({1.0/avg_time:.1f} FPS)" if avg_time > 0 else "No frames processed")
            print(f"Processed {frame_idx} frames")
            
            # Print tracking stats (with error handling)
            try:
                reid_stats = tracker.get_reid_stats()
                print(f"Re-ID Statistics: {reid_stats['total_reid_attempts']} attempts, "
                    f"{reid_stats['identity_confirmations']} confirmations, "
                    f"{reid_stats['identity_switches']} switches, "
                    f"{reid_stats['newly_identified']} newly identified")
            except Exception as e:
                print(f"Error getting Re-ID stats: {e}")
                
            # Print time data for tracked people (with error handling)
            try:
                time_data = tracker.get_time_data()
                print("\nTracked People:")
                for person in time_data:
                    if person.get('is_known', False):
                        print(f"  {person['name']}: {person['duration']} (first seen: {person['first_seen']}, last seen: {person['last_seen']})")
                    else:
                        print(f"  Person #{person.get('id', 'unknown')}: {person['duration']} (first seen: {person['first_seen']}, last seen: {person['last_seen']})")
            except Exception as e:
                print(f"Error getting time data: {e}")
        except Exception as e:
            print(f"Error printing final stats: {e}")

def draw_results(frame, tracked_objects, tracker, current_time):
    """
    Draw bounding boxes and labels for tracked objects
    
    Args:
        frame: Input video frame
        tracked_objects: Dict of tracked objects from PersonTracker
        tracker: SafePersonTracker instance for reid timing info
        current_time: Current timestamp
    
    Returns:
        Frame with visualization
    """
    result = frame.copy()
    
    # Draw all tracked objects
    for obj_id, obj_data in tracked_objects.items():
        if not obj_data.get('active', True):
            continue  # Don't draw inactive tracks
        
        try:
            x1, y1, x2, y2 = map(int, obj_data['bbox'])
            name = obj_data.get('name', 'UNK')
            
            # Calculate center of box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Determine color based on identification
            if isinstance(obj_id, str):
                # Temporary track - yellow
                color = (0, 255, 255)  # Yellow in BGR
                label = f"Temp #{obj_id}"
                id_text = f"T{obj_id}"
            elif name == "UNK":
                # Unknown person - orange
                color = (0, 165, 255)  # Orange in BGR
                label = f"#{obj_id}"
                id_text = f"#{obj_id}"
            else:
                # Known person - green
                color = (0, 255, 0)  # Green in BGR
                label = f"{name}"
                id_text = f"{name}"
            
            # Draw box with thickness based on size
            thickness = max(1, min(3, int((x2-x1) / 200 + 1)))
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Add text background at top
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(result, 
                        (x1, y1 - text_size[1] - 10), 
                        (x1 + text_size[0] + 10, y1), 
                        color, -1)
            
            # Draw label text at top
            cv2.putText(result, label, (x1 + 5, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Draw ID in center of box with larger font
            id_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            
            # Add white background for contrast
            bg_margin = 5
            cv2.rectangle(result, 
                        (center_x - id_size[0]//2 - bg_margin, center_y - id_size[1]//2 - bg_margin), 
                        (center_x + id_size[0]//2 + bg_margin, center_y + id_size[1]//2 + bg_margin), 
                        (255, 255, 255), -1)
            
            # Draw ID text with black outline for better visibility
            cv2.putText(result, id_text, (center_x - id_size[0]//2, center_y + id_size[1]//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                    
            # Add re-identification indicator
            if not isinstance(obj_id, str):  # Skip for temporary tracks
                try:
                    # Calculate time until next re-ID
                    time_until_reid = tracker.get_next_reid_time(obj_id, current_time)
                    
                    # Convert to percentage (0-100%)
                    reid_progress = 100 - min(100, int((time_until_reid / tracker.periodic_reid_interval) * 100))
                    
                    # Draw re-ID progress bar
                    bar_width = x2 - x1
                    bar_height = 5
                    bar_y = y2 + 10
                    
                    # Draw background (gray)
                    cv2.rectangle(result, (x1, bar_y), (x2, bar_y + bar_height), (80, 80, 80), -1)
                    
                    # Draw progress (color based on how close to next re-ID)
                    if reid_progress < 30:
                        bar_color = (0, 0, 200)  # Red (far from next re-ID)
                    elif reid_progress < 70:
                        bar_color = (0, 200, 200)  # Yellow (approaching re-ID)
                    else:
                        bar_color = (0, 200, 0)  # Green (close to re-ID)
                        
                    # Calculate progress width
                    progress_width = int(bar_width * reid_progress / 100)
                    if progress_width > 0:
                        cv2.rectangle(result, (x1, bar_y), (x1 + progress_width, bar_y + bar_height), 
                                    bar_color, -1)
                    
                    # Add a visual indicator if re-ID was recently performed (within last 0.5 seconds)
                    if obj_id in tracker.reid_attempts:
                        last_reid = tracker.reid_attempts.get(obj_id, 0)
                        if current_time - last_reid < 0.5:  # Show indicator for 0.5 seconds
                            # Draw re-ID indicator (pulsing circle)
                            cv2.circle(result, (x2 + 15, y1 + 15), 8, (0, 0, 255), -1)  # Red circle
                            cv2.putText(result, "Re-ID", (x2 + 25, y1 + 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                except Exception as e:
                    # Skip re-ID visualization if there's an error
                    pass
            
        except Exception as e:
            # Skip this object if there's an error drawing it
            print(f"Error drawing object {obj_id}: {e}")
            continue
    
    return result

if __name__ == '__main__':
    # Set default parameters if no arguments are provided
    video_path = "test.mp4"  # Default video path
    output_path = "out.mp4"  # Default output path
    skip_frames = 0  # Process every frame
    confidence = 0.75  # Default confidence threshold
    headless = True  # Default to try using GUI
    reid_interval = 5  # Default re-ID interval in seconds
    
    # Check for models
    if model is None:
        print("Error: YOLO model not available. Make sure 'models/yolov8n.pt' exists.")
    if face_model is None:
        print("Error: ArcFace model not available. Make sure insightface is installed.")
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    # Check for headless flag in arguments
    if len(sys.argv) > 3:
        arg = sys.argv[3].lower()
        if arg in ["--headless", "-h", "headless"]:
            headless = True
            print("Running in headless mode (no GUI display)")
        elif arg.startswith("--reid=") or arg.startswith("reid="):
            # Parse reid interval (--reid=3.0 or reid=3.0)
            try:
                reid_interval = float(arg.split("=")[1])
                print(f"Re-ID interval set to {reid_interval} seconds")
            except:
                print(f"Invalid re-ID interval format: {arg}, using default {reid_interval}s")
    
    # Check for additional args in 4th position
    if len(sys.argv) > 4:
        arg = sys.argv[4].lower()
        if arg.startswith("--reid=") or arg.startswith("reid="):
            try:
                reid_interval = float(arg.split("=")[1])
                print(f"Re-ID interval set to {reid_interval} seconds")
            except:
                print(f"Invalid re-ID interval format: {arg}, using default {reid_interval}s")
    
    print(f"Processing video: {video_path}")
    print(f"Output path: {output_path}")
    
    # Process the video
    process_video(video_path, output_path, skip_frames, confidence, headless, reid_interval) 