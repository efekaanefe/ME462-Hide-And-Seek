#!/usr/bin/env python3
"""
Utility script to set up the rooms database structure
"""

import os
import argparse
import shutil
from pathlib import Path

def create_directory_structure(base_dir, num_rooms, cameras_per_room):
    """
    Create the directory structure for rooms and cameras
    
    Args:
        base_dir: Base directory where 'rooms_database' will be created
        num_rooms: Number of rooms to create
        cameras_per_room: Number of cameras per room
    """
    # Create base directory
    db_path = os.path.join(base_dir, "rooms_database")
    os.makedirs(db_path, exist_ok=True)
    
    for room_idx in range(num_rooms):
        room_dir = os.path.join(db_path, f"room{room_idx}")
        os.makedirs(room_dir, exist_ok=True)
        
        # Create a placeholder for the 2D map
        map_placeholder = os.path.join(room_dir, "2Dmap.png")
        if not os.path.exists(map_placeholder):
            print(f"Please place a 2D map image at: {map_placeholder}")
        
        # Create camera directories
        for cam_idx in range(cameras_per_room):
            cam_dir = os.path.join(room_dir, f"cam{cam_idx}")
            os.makedirs(cam_dir, exist_ok=True)
            
            # Create a README to explain how to add images
            readme_path = os.path.join(cam_dir, "README.txt")
            with open(readme_path, 'w') as f:
                f.write(f"Place camera {cam_idx} images for room {room_idx} in this directory.\n")
                f.write("The images should be in PNG, JPG, or JPEG format.\n")
    
    print(f"Directory structure created at {db_path}")
    print("Please add your map images and camera images to the respective directories.")

def copy_images_to_database(base_dir, room_idx, cam_idx, source_images):
    """
    Copy images to the specified camera directory
    
    Args:
        base_dir: Base directory containing 'rooms_database'
        room_idx: Room index
        cam_idx: Camera index
        source_images: List of source image paths
    """
    target_dir = os.path.join(base_dir, "rooms_database", f"room{room_idx}", f"cam{cam_idx}")
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    
    for img_path in source_images:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"Warning: Image {img_path} not found")
            continue
            
        if not img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            print(f"Warning: {img_path} is not a supported image format")
            continue
            
        target_path = os.path.join(target_dir, img_path.name)
        shutil.copy2(img_path, target_path)
        print(f"Copied {img_path} to {target_path}")

def copy_map_to_database(base_dir, room_idx, map_path):
    """
    Copy a map image to the specified room directory
    
    Args:
        base_dir: Base directory containing 'rooms_database'
        room_idx: Room index
        map_path: Path to the map image
    """
    map_path = Path(map_path)
    if not map_path.exists():
        print(f"Error: Map image {map_path} not found")
        return False
        
    if not map_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
        print(f"Error: {map_path} is not a supported image format")
        return False
        
    target_dir = os.path.join(base_dir, "rooms_database", f"room{room_idx}")
    os.makedirs(target_dir, exist_ok=True)
    
    target_path = os.path.join(target_dir, "2Dmap.png")
    shutil.copy2(map_path, target_path)
    print(f"Copied map image to {target_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Set up rooms database structure")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create structure command
    create_parser = subparsers.add_parser("create", help="Create database directory structure")
    create_parser.add_argument("--base-dir", type=str, default=".", help="Base directory for the database")
    create_parser.add_argument("--rooms", type=int, default=1, help="Number of rooms to create")
    create_parser.add_argument("--cameras", type=int, default=1, help="Number of cameras per room")
    
    # Add camera images command
    add_cam_parser = subparsers.add_parser("add-camera", help="Add camera images to the database")
    add_cam_parser.add_argument("--base-dir", type=str, default=".", help="Base directory containing the database")
    add_cam_parser.add_argument("--room", type=int, required=True, help="Room index")
    add_cam_parser.add_argument("--camera", type=int, required=True, help="Camera index")
    add_cam_parser.add_argument("--images", type=str, nargs="+", required=True, help="Paths to camera images")
    
    # Add map image command
    add_map_parser = subparsers.add_parser("add-map", help="Add a map image to the database")
    add_map_parser.add_argument("--base-dir", type=str, default=".", help="Base directory containing the database")
    add_map_parser.add_argument("--room", type=int, required=True, help="Room index")
    add_map_parser.add_argument("--map", type=str, required=True, help="Path to map image")
    
    args = parser.parse_args()
    
    if args.command == "create":
        create_directory_structure(args.base_dir, args.rooms, args.cameras)
    elif args.command == "add-camera":
        copy_images_to_database(args.base_dir, args.room, args.camera, args.images)
    elif args.command == "add-map":
        copy_map_to_database(args.base_dir, args.room, args.map)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 