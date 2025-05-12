#!/usr/bin/env python3
"""
Demo script showing how to use the HomographyTool programmatically
"""

from homography_modular import HomographyTool

def visualize_saved_homographies():
    """
    Load and visualize previously saved homography matrices
    """
    tool = HomographyTool()
    
    # Load previously saved homography matrices
    tool.load_homography_matrices()
    
    # Show available homographies
    print("Available homographies:")
    for key in tool.homography_matrices.keys():
        print(f"- {key}")
    
    # Select a homography to visualize
    room_idx = int(input("Enter room index: "))
    cam_idx = int(input("Enter camera index: "))
    
    # Visualize the selected homography
    tool.visualize_homography(room_idx, cam_idx)

def create_homography_for_new_camera(room_idx, cam_idx):
    """
    Create a new homography for a specific room and camera
    """
    tool = HomographyTool()
    
    # Load any existing homography matrices
    try:
        tool.load_homography_matrices()
        print("Loaded existing homography matrices")
    except:
        print("No existing homography matrices found, starting fresh")
    
    # Select room and camera
    tool.select_room(room_idx)
    tool.select_camera(cam_idx)
    
    # Select points
    print("\nSelect corresponding points on the camera and map images:")
    print("1. First click on the camera image")
    print("2. Then click on the corresponding point on the map image")
    print("3. Repeat until you have enough points (at least 4)")
    print("4. Close the window when done")
    
    tool.select_points()
    
    # Calculate homography
    H = tool.calculate_homography()
    if H is not None:
        print(f"Successfully calculated homography matrix for room{room_idx}/cam{cam_idx}")
        
        # Save the matrices
        tool.save_homography_matrices()
        
        # Visualize the result
        tool.visualize_homography(room_idx, cam_idx)
    else:
        print("Failed to calculate homography matrix")

def add_new_points_to_existing_homography(room_idx, cam_idx):
    """
    Add more points to an existing homography to improve it
    """
    tool = HomographyTool()
    
    # Load existing homography matrices
    tool.load_homography_matrices()
    
    key = f"room{room_idx}_cam{cam_idx}"
    if key not in tool.homography_matrices:
        print(f"No existing homography found for {key}")
        return
    
    # Load the existing points
    tool.select_room(room_idx)
    tool.select_camera(cam_idx)
    
    data = tool.homography_matrices[key]
    tool.cam_points = data["camera_points"]
    tool.map_points = data["map_points"]
    tool.point_labels = data.get("point_labels", [f"Point {i+1}" for i in range(len(tool.cam_points))])
    
    print(f"Loaded {len(tool.cam_points)} existing points")
    
    # Select additional points
    print("\nSelect additional corresponding points:")
    print("1. First click on the camera image")
    print("2. Then click on the corresponding point on the map image")
    print("3. Repeat until you have enough points")
    print("4. Close the window when done")
    
    tool.select_points()
    
    # Calculate new homography
    H = tool.calculate_homography()
    if H is not None:
        print(f"Successfully updated homography matrix for {key}")
        
        # Save the matrices
        tool.save_homography_matrices()
        
        # Visualize the result
        tool.visualize_homography(room_idx, cam_idx)
    else:
        print("Failed to update homography matrix")

def main():
    print("Homography Tool Demo")
    print("====================\n")
    
    print("1. Visualize saved homographies")
    print("2. Create homography for a new camera")
    print("3. Add points to an existing homography")
    
    choice = int(input("\nSelect an option: "))
    
    if choice == 1:
        visualize_saved_homographies()
    elif choice == 2:
        room_idx = int(input("Enter room index: "))
        cam_idx = int(input("Enter camera index: "))
        create_homography_for_new_camera(room_idx, cam_idx)
    elif choice == 3:
        room_idx = int(input("Enter room index: "))
        cam_idx = int(input("Enter camera index: "))
        add_new_points_to_existing_homography(room_idx, cam_idx)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main() 