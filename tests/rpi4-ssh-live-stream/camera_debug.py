#!/usr/bin/env python3
import cv2
import sys

def test_camera(index):
    print(f"\n=== Testing Camera {index} ===")
    cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print(f"Camera {index}: Cannot open")
        return False
    
    # Get camera properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera {index}: Default resolution {int(width)}x{int(height)}, FPS: {fps}")
    
    # Try to capture a frame
    ret, frame = cap.read()
    if ret and frame is not None:
        h, w = frame.shape[:2]
        print(f"Camera {index}: Successfully captured frame {w}x{h}")
        print(f"Camera {index}: Frame shape: {frame.shape}, dtype: {frame.dtype}")
        
        # Test encoding
        try:
            ret_encode, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret_encode:
                print(f"Camera {index}: JPEG encoding successful, size: {len(buffer)} bytes")
            else:
                print(f"Camera {index}: JPEG encoding failed")
        except Exception as e:
            print(f"Camera {index}: JPEG encoding error: {e}")
            
        cap.release()
        return True
    else:
        print(f"Camera {index}: Failed to capture frame")
        cap.release()
        return False

def main():
    print("OpenCV Camera Debug Tool")
    print(f"OpenCV Version: {cv2.__version__}")
    
    working_cameras = []
    
    # Test cameras 0-3
    for i in range(4):
        if test_camera(i):
            working_cameras.append(i)
    
    print(f"\n=== Summary ===")
    if working_cameras:
        print(f"Working cameras: {working_cameras}")
        print(f"Recommended camera index: {working_cameras[0]}")
    else:
        print("No working cameras found!")
        print("Check:")
        print("1. Camera is connected")
        print("2. Camera permissions (try: sudo usermod -a -G video $USER)")
        print("3. Camera is not being used by another application")

if __name__ == "__main__":
    main()
