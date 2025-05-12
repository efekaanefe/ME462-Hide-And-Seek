#!/usr/bin/env python3
"""
Demo script showing how to use the HomographyTool programmatically
"""

from homography_modular import HomographyTool
import os

def visualize_saved_homographies():
    """
    Load and visualize previously saved homography matrices
    """
    tool = HomographyTool()
    
    # Load previously saved homography matrices
    try:
        tool.load_homography_matrices()
        
        # Show available homographies
        print("\nAvailable homographies:")
        for key in tool.homography_matrices.keys():
            print(f"- {key}")
        
        if not tool.homography_matrices:
            print("No saved homography matrices found.")
            return
            
        # Select a homography to visualize
        room_idx = int(input("\nEnter room index: "))
        cam_idx = int(input("Enter camera index: "))
        
        # Visualize the selected homography
        tool.visualize_homography(room_idx, cam_idx)
    except Exception as e:
        print(f"Error: {str(e)}")

def create_homography_for_new_camera(room_idx, cam_idx):
    """
    Create a new homography for a specific room and camera
    """
    tool = HomographyTool()
    
    # Load any existing homography matrices
    try:
        if os.path.exists("homography_matrices.json"):
            tool.load_homography_matrices()
            print("Loaded existing homography matrices")
            
            # Check if this room/camera already has a homography
            key = f"room{room_idx}_cam{cam_idx}"
            if key in tool.homography_matrices:
                print(f"\nWarning: A homography for {key} already exists.")
                overwrite = input("Do you want to overwrite it? (y/n): ")
                if overwrite.lower() != 'y':
                    print("Operation canceled")
                    return
                print(f"Overwriting existing homography for {key}")
    except Exception as e:
        print(f"Warning: {str(e)}")
        print("Starting with a fresh homography matrix set")
    
    # Select room and camera
    try:
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
            print(f"\nSuccessfully calculated homography matrix for room{room_idx}/cam{cam_idx}")
            
            # Save the matrices
            tool.save_homography_matrices()
            
            # Ask if the user wants to visualize the result
            visualize = input("\nDo you want to visualize the homography? (y/n): ")
            if visualize.lower() == 'y':
                tool.visualize_homography(room_idx, cam_idx)
        else:
            print("Failed to calculate homography matrix")
    except Exception as e:
        print(f"Error: {str(e)}")

def add_new_points_to_existing_homography(room_idx, cam_idx):
    """
    Add more points to an existing homography to improve it
    """
    tool = HomographyTool()
    
    # Load existing homography matrices
    try:
        tool.load_homography_matrices()
        
        key = f"room{room_idx}_cam{cam_idx}"
        if key not in tool.homography_matrices:
            print(f"No existing homography found for {key}")
            create_new = input("Do you want to create a new homography instead? (y/n): ")
            if create_new.lower() == 'y':
                create_homography_for_new_camera(room_idx, cam_idx)
            return
        
        # Load the existing points
        tool.select_room(room_idx)
        tool.select_camera(cam_idx)
        
        data = tool.homography_matrices[key]
        tool.cam_points = data["camera_points"]
        tool.map_points = data["map_points"]
        tool.point_labels = data.get("point_labels", [f"Point {i+1}" for i in range(len(tool.cam_points))])
        
        print(f"\nLoaded {len(tool.cam_points)} existing points")
        
        # Select additional points
        print("\nSelect additional corresponding points:")
        print("1. First click on the camera image")
        print("2. Then click on the corresponding point on the map image")
        print("3. Repeat until you have enough additional points")
        print("4. Close the window when done")
        
        tool.select_points()
        
        # Calculate new homography
        H = tool.calculate_homography()
        if H is not None:
            print(f"\nSuccessfully updated homography matrix for {key}")
            
            # Save the matrices
            tool.save_homography_matrices()
            
            # Ask if the user wants to visualize the result
            visualize = input("\nDo you want to visualize the updated homography? (y/n): ")
            if visualize.lower() == 'y':
                tool.visualize_homography(room_idx, cam_idx)
        else:
            print("Failed to update homography matrix")
    except Exception as e:
        print(f"Error: {str(e)}")

def delete_homography(room_idx, cam_idx):
    """
    Delete a specific homography matrix
    """
    tool = HomographyTool()
    
    # Load existing homography matrices
    try:
        tool.load_homography_matrices()
        
        key = f"room{room_idx}_cam{cam_idx}"
        if key not in tool.homography_matrices:
            print(f"No homography found for {key}")
            return
        
        # Confirm deletion
        confirm = input(f"Are you sure you want to delete the homography for {key}? (y/n): ")
        if confirm.lower() != 'y':
            print("Deletion canceled")
            return
        
        # Delete the homography
        del tool.homography_matrices[key]
        
        # Save the updated matrices
        tool.save_homography_matrices()
        print(f"Homography for {key} has been deleted")
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    print("Homography Tool Demo")
    print("====================\n")
    
    print("1. Visualize saved homographies")
    print("2. Create homography for a new camera")
    print("3. Add points to an existing homography")
    print("4. Delete a homography")
    
    try:
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
        elif choice == 4:
            room_idx = int(input("Enter room index: "))
            cam_idx = int(input("Enter camera index: "))
            delete_homography(room_idx, cam_idx)
        else:
            print("Invalid choice")
    except ValueError:
        print("Please enter a valid number")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 