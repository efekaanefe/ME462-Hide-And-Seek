import cv2
import numpy as np
import time
import argparse
from ultralytics import YOLO

class PeopleBEVMapper:
    def __init__(self, input_video, output_video=None, calibration_mode=False):
        self.input_video = input_video
        self.output_video = output_video
        self.calibration_mode = calibration_mode
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(input_video)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {input_video}")
        
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize video writer if output is specified
        self.out = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(output_video, fourcc, self.fps, 
                                      (self.frame_width, self.frame_height + 400))  # Extra space for BEV
        
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')  # Using YOLOv8 nano
        
        # Default homography points (will be updated during calibration)
        # Source points in camera view
        self.src_points = np.array([
            [100, self.frame_height - 50],  # Bottom left
            [self.frame_width - 100, self.frame_height - 50],  # Bottom right
            [self.frame_width - 200, int(self.frame_height * 0.4)],  # Top right
            [200, int(self.frame_height * 0.4)]  # Top left
        ], dtype=np.float32)
        
        # Destination points in BEV (assuming 400x400 BEV image)
        self.bev_width, self.bev_height = 400, 400
        self.dst_points = np.array([
            [50, self.bev_height - 50],  # Bottom left
            [self.bev_width - 50, self.bev_height - 50],  # Bottom right
            [self.bev_width - 50, 50],  # Top right
            [50, 50]  # Top left
        ], dtype=np.float32)
        
        # Homography matrix
        self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        
        # Create BEV image (bird's eye view)
        self.bev_image = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
        
        # For calibration
        self.current_point = 0
        self.calibrating = False
    
    def calibrate(self):
        """Run calibration mode to define homography points"""
        self.calibrating = True
        print("Calibration Mode:")
        print("Click on 4 points in this order: Bottom-Left, Bottom-Right, Top-Right, Top-Left")
        print("Press 'r' to reset points, 'c' to confirm and continue")
        
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        
        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', self.mouse_callback)
        
        while self.calibrating:
            display_frame = frame.copy()
            
            # Draw points and lines
            for i, point in enumerate(self.src_points):
                cv2.circle(display_frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
                if i > 0:
                    cv2.line(display_frame, 
                             (int(self.src_points[i-1][0]), int(self.src_points[i-1][1])),
                             (int(point[0]), int(point[1])),
                             (0, 255, 0), 2)
            
            # Connect last and first point
            if len(self.src_points) == 4:
                cv2.line(display_frame, 
                         (int(self.src_points[3][0]), int(self.src_points[3][1])),
                         (int(self.src_points[0][0]), int(self.src_points[0][1])),
                         (0, 255, 0), 2)
            
            cv2.putText(display_frame, f"Point {self.current_point+1}/4", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow('Calibration', display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # Reset points
                self.current_point = 0
                print("Points reset")
            elif key == ord('c') and self.current_point == 4:  # Confirm and continue
                self.calibrating = False
                self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
                print("Calibration confirmed")
                break
            elif key == 27:  # ESC to exit
                self.calibrating = False
                print("Calibration canceled")
                break
        
        cv2.destroyWindow('Calibration')
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for calibration"""
        if event == cv2.EVENT_LBUTTONDOWN and self.current_point < 4:
            self.src_points[self.current_point] = [x, y]
            self.current_point += 1
            print(f"Point {self.current_point} set at ({x}, {y})")
    
    def detect_people(self, frame):
        """Detect people in the frame using YOLO"""
        results = self.model(frame, classes=0)  # class 0 is person in COCO dataset
        
        # Extract bounding boxes
        boxes = []
        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # Person class
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    if confidence > 0.5:  # Confidence threshold
                        # Calculate bottom center point (feet position)
                        foot_point = (int((x1 + x2) / 2), y2)
                        boxes.append((x1, y1, x2, y2, foot_point))
        
        return boxes
    
    def map_to_bev(self, foot_points):
        """Map detected foot points to bird's eye view using homography"""
        bev_points = []
        
        for point in foot_points:
            # Convert to homogeneous coordinates
            p = np.array([[[point[0], point[1]]]], dtype=np.float32)
            
            # Apply homography
            p_transformed = cv2.perspectiveTransform(p, self.H)[0, 0]
            bev_points.append((int(p_transformed[0]), int(p_transformed[1])))
        
        return bev_points
    
    def visualize_results(self, frame, boxes, bev_points):
        """Visualize results with bounding boxes and BEV mapping"""
        # Clear BEV image
        self.bev_image.fill(0)
        
        # Draw ground plane grid on BEV
        grid_size = 50
        for i in range(0, self.bev_width, grid_size):
            cv2.line(self.bev_image, (i, 0), (i, self.bev_height), (30, 30, 30), 1)
        for i in range(0, self.bev_height, grid_size):
            cv2.line(self.bev_image, (0, i), (self.bev_width, i), (30, 30, 30), 1)
        
        # Draw calibration region on BEV
        cv2.polylines(self.bev_image, [np.int32(self.dst_points)], True, (0, 255, 0), 2)
        
        # Draw detected people on original frame
        for i, (x1, y1, x2, y2, foot_point) in enumerate(boxes):
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw foot point
            cv2.circle(frame, foot_point, 5, (0, 0, 255), -1)
            
            # Label with person ID
            cv2.putText(frame, f"Person {i+1}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw calibration region on original frame
        cv2.polylines(frame, [np.int32(self.src_points)], True, (0, 255, 0), 2)
        
        # Draw detected people on BEV
        for i, point in enumerate(bev_points):
            # Draw person as circle
            cv2.circle(self.bev_image, point, 10, (0, 0, 255), -1)
            
            # Label with person ID
            cv2.putText(self.bev_image, f"{i+1}", (point[0]+5, point[1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Combine original frame and BEV
        combined_frame = np.vstack([frame, 
                                   cv2.resize(self.bev_image, (self.frame_width, 400))])
        
        return combined_frame
    
    def process(self):
        """Main processing loop"""
        # Run calibration if needed
        if self.calibration_mode:
            self.calibrate()
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect people and get bounding boxes
            boxes = self.detect_people(frame)
            
            # Extract foot points
            foot_points = [box[4] for box in boxes]
            
            # Map foot points to BEV
            bev_points = self.map_to_bev(foot_points)
            
            # Visualize results
            result_frame = self.visualize_results(frame, boxes, bev_points)
            
            # Display FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Display people count
            cv2.putText(result_frame, f"People: {len(boxes)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Show result
            cv2.imshow('People BEV Mapping', result_frame)
            
            # Write frame to output video if specified
            if self.out is not None:
                self.out.write(result_frame)
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
        
        # Clean up
        self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='People BEV Mapping from Video')
    parser.add_argument('--input', type=str, required=True, help='Input video file')
    parser.add_argument('--output', type=str, help='Output video file (optional)')
    parser.add_argument('--calibrate', action='store_true', help='Run calibration mode')
    
    args = parser.parse_args()
    
    mapper = PeopleBEVMapper(args.input, args.output, args.calibrate)
    mapper.process()

if __name__ == "__main__":
    main()
    # python yolo_test.py --input test.mp4 --output out.mp4 --calibrate
