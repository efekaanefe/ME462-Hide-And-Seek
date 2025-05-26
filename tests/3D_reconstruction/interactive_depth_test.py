import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        depth_value = param[y, x]
        # Create a copy of the image to draw on
        display_img = cv2.cvtColor(param, cv2.COLOR_GRAY2BGR)
        # Draw a crosshair at cursor position
        cv2.drawMarker(display_img, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        # Add depth value text
        cv2.putText(display_img, f"Depth: {depth_value:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Depth Image', display_img)

def main():
    # Read the depth image
    depth_img = cv2.imread('test_person_depth.png', cv2.IMREAD_GRAYSCALE)
    # Create window and set mouse callback
    cv2.namedWindow('Depth Image')
    cv2.setMouseCallback('Depth Image', mouse_callback, depth_img)

    # Display initial image
    cv2.imshow('Depth Image', depth_img)

    print("Move your mouse over the image to see depth values")
    print("Press 'q' to quit")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
