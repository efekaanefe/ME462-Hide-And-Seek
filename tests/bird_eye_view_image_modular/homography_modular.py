import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.patches import Circle
from typing import List, Dict, Tuple, Optional

class HomographyTool:
    def __init__(self, database_path: str = "rooms_database"):
        """
        Initialize the homography tool with the room database path.
        
        Args:
            database_path: Path to the rooms database directory
        """
        self.database_path = database_path
        self.room_paths = self._get_room_paths()
        self.selected_room = None
        self.selected_cam = None
        self.cam_image = None
        self.map_image = None
        self.cam_points = []
        self.map_points = []
        self.homography_matrices = {}
        self.point_labels = []
        self.waiting_for_map_point = False  # Flag to track if we're waiting for a map point
        self.current_point_num = 0  # Track current point number
        self.cam_ax = None  # Camera axis reference
        self.map_ax = None  # Map axis reference
        
    def _get_room_paths(self) -> List[str]:
        """Get list of room directories"""
        if not os.path.exists(self.database_path):
            raise FileNotFoundError(f"Database path {self.database_path} not found")
        
        return [d for d in os.listdir(self.database_path) 
                if os.path.isdir(os.path.join(self.database_path, d)) and d.startswith("room")]
    
    def select_room(self, room_index: int) -> None:
        """
        Select a room by index
        
        Args:
            room_index: Index of the room to select
        """
        room_name = f"room{room_index}"
        if room_name not in self.room_paths:
            raise ValueError(f"Room {room_name} not found in database")
        
        self.selected_room = room_name
        self.map_image = self._load_map_image()
        print(f"Selected room: {self.selected_room}")
        print(f"Map image shape: {self.map_image.shape}")
    
    def _load_map_image(self) -> np.ndarray:
        """Load the map image for the selected room"""
        map_path = os.path.join(self.database_path, self.selected_room, "2Dmap.png")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Map image not found at {map_path}")
        
        map_img = cv2.imread(map_path)
        if map_img is None:
            raise ValueError(f"Failed to load map image from {map_path}")
            
        return cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
    
    def get_cameras_for_room(self) -> List[str]:
        """Get list of cameras for the selected room"""
        if not self.selected_room:
            raise ValueError("No room selected")
            
        room_path = os.path.join(self.database_path, self.selected_room)
        return [d for d in os.listdir(room_path) 
                if os.path.isdir(os.path.join(room_path, d)) and d.startswith("cam")]
    
    def select_camera(self, cam_index: int) -> None:
        """
        Select a camera by index
        
        Args:
            cam_index: Index of the camera to select
        """
        cam_name = f"cam{cam_index}"
        cameras = self.get_cameras_for_room()
        if cam_name not in cameras:
            raise ValueError(f"Camera {cam_name} not found in room {self.selected_room}")
        
        self.selected_cam = cam_name
        self.cam_image = self._load_camera_image()
        # Reset points when changing camera
        self.cam_points = []
        self.map_points = []
        self.point_labels = []
        self.waiting_for_map_point = False
        self.current_point_num = 0
        print(f"Selected camera: {self.selected_cam}")
        print(f"Camera image shape: {self.cam_image.shape}")
    
    def _load_camera_image(self) -> np.ndarray:
        """Load the camera image for the selected camera"""
        cam_dir = os.path.join(self.database_path, self.selected_room, self.selected_cam)
        if not os.path.exists(cam_dir):
            raise FileNotFoundError(f"Camera directory not found at {cam_dir}")
            
        # Find the first image file in the camera directory
        image_files = [f for f in os.listdir(cam_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            raise FileNotFoundError(f"No image files found in {cam_dir}")
            
        cam_img = cv2.imread(os.path.join(cam_dir, image_files[0]))
        if cam_img is None:
            raise ValueError(f"Failed to load camera image from {os.path.join(cam_dir, image_files[0])}")
            
        return cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
    
    def _on_click(self, event) -> None:
        """Master click handler that routes to the correct handler based on the axes"""
        if event.xdata is None or event.ydata is None:
            return
            
        # Check which subplot was clicked
        if event.inaxes == self.cam_ax:
            self._handle_camera_click(event)
        elif event.inaxes == self.map_ax:
            self._handle_map_click(event)
    
    def _handle_camera_click(self, event) -> None:
        """Handle click events on the camera image"""
        if self.waiting_for_map_point:
            print("Please select a point on the map first")
            return
        
        x, y = int(event.xdata), int(event.ydata)
        self.cam_points.append([x, y])
        
        # Update point label and current point number
        self.current_point_num = len(self.cam_points)
        self.point_labels.append(f"Point {self.current_point_num}")
        
        # Draw the point
        circle = Circle((x, y), 5, color='red', fill=True)
        self.cam_ax.add_patch(circle)
        self.cam_ax.text(x + 10, y, f"Point {self.current_point_num}", color='red', fontsize=12)
        plt.draw()
        
        print(f"Selected camera point {self.current_point_num}: ({x}, {y})")
        print("Now select the corresponding point on the map image")
        
        # Set flag to indicate we're waiting for a map point
        self.waiting_for_map_point = True
    
    def _handle_map_click(self, event) -> None:
        """Handle click events on the map image"""
        # Only accept map clicks if we're waiting for a map point
        if not self.waiting_for_map_point:
            print("Please select a point on the camera image first")
            return
        
        x, y = int(event.xdata), int(event.ydata)
        self.map_points.append([x, y])
        
        # Draw the point
        circle = Circle((x, y), 5, color='red', fill=True)
        self.map_ax.add_patch(circle)
        self.map_ax.text(x + 10, y, f"Point {self.current_point_num}", color='red', fontsize=12)
        plt.draw()
        
        print(f"Selected map point {self.current_point_num}: ({x}, {y})")
        print(f"Point {self.current_point_num} selected on both images")
        
        # Reset flag to allow selecting next camera point
        self.waiting_for_map_point = False
    
    def select_points(self) -> None:
        """Open interactive windows to select corresponding points on camera and map images"""
        if self.cam_image is None or self.map_image is None:
            raise ValueError("Camera and map images must be loaded first")
        
        # Reset the waiting flag at the start of point selection
        self.waiting_for_map_point = False
        
        # Create figure with two subplots
        fig, (self.cam_ax, self.map_ax) = plt.subplots(1, 2, figsize=(14, 7))
        fig.canvas.manager.set_window_title(f"Point Selection: {self.selected_room}/{self.selected_cam}")
        
        # Camera image
        self.cam_ax.imshow(self.cam_image)
        self.cam_ax.set_title(f"Camera Image ({self.selected_cam})")
        self.cam_ax.axis('on')
        
        # Map image
        self.map_ax.imshow(self.map_image)
        self.map_ax.set_title("Map Image")
        self.map_ax.axis('on')
        
        # Status text for user feedback
        status_text = fig.text(0.5, 0.01, 
                "Instructions: First click on the camera image, then on the corresponding map point. Repeat.",
                ha='center', fontsize=12, bbox=dict(facecolor='lightgray', alpha=0.5))
        
        # Set up click handler
        cid = fig.canvas.mpl_connect('button_press_event', self._on_click)
        
        # Update plot layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)  # Make room for the instructions
        
        # Show already selected points if any
        self._display_existing_points()
        
        # Show the figure
        plt.show()
        
        # Disconnect the event handler when plot is closed
        fig.canvas.mpl_disconnect(cid)
    
    def _display_existing_points(self) -> None:
        """Display any existing points that were previously selected"""
        for i, (cam_pt, map_pt) in enumerate(zip(self.cam_points, self.map_points)):
            point_num = i + 1
            
            # Camera point
            circle = Circle((cam_pt[0], cam_pt[1]), 5, color='red', fill=True)
            self.cam_ax.add_patch(circle)
            self.cam_ax.text(cam_pt[0] + 10, cam_pt[1], f"Point {point_num}", color='red', fontsize=12)
            
            # Map point
            circle = Circle((map_pt[0], map_pt[1]), 5, color='red', fill=True)
            self.map_ax.add_patch(circle)
            self.map_ax.text(map_pt[0] + 10, map_pt[1], f"Point {point_num}", color='red', fontsize=12)
    
    def calculate_homography(self) -> Optional[np.ndarray]:
        """
        Calculate the homography matrix using the selected points
        
        Returns:
            Homography matrix or None if insufficient points
        """
        if len(self.cam_points) < 4 or len(self.map_points) < 4:
            print("Need at least 4 point pairs to calculate homography")
            return None
        
        if len(self.cam_points) != len(self.map_points):
            print("Number of camera points must match number of map points")
            return None
        
        # Convert points to numpy arrays
        src_pts = np.array(self.cam_points, dtype=np.float32)
        dst_pts = np.array(self.map_points, dtype=np.float32)
        
        # Calculate homography matrix
        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Store the homography matrix for current room and camera
        key = f"{self.selected_room}_{self.selected_cam}"
        self.homography_matrices[key] = {
            "matrix": H.tolist(),
            "camera_points": self.cam_points,
            "map_points": self.map_points,
            "point_labels": self.point_labels
        }
        
        print(f"Homography matrix calculated for {key}")
        return H
    
    def save_homography_matrices(self, output_file: str = "homography_matrices.json") -> None:
        """
        Save all calculated homography matrices to a JSON file
        
        Args:
            output_file: Path to save the JSON file
            
        This method will:
        1. Load existing matrices from the file if it exists
        2. Update the loaded data with any new matrices (overwriting existing ones with same keys)
        3. Write the merged data back to the file
        """
        if not self.homography_matrices:
            print("No homography matrices to save")
            return
        
        # Load existing matrices if the file exists
        existing_matrices = {}
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    existing_matrices = json.load(f)
                print(f"Loaded existing matrices from {output_file}")
            except json.JSONDecodeError:
                print(f"Warning: Existing file {output_file} has invalid JSON format. Creating new file.")
            except Exception as e:
                print(f"Warning: Could not read existing file {output_file}: {str(e)}. Creating new file.")
        
        # Merge existing matrices with new ones (new ones take precedence)
        merged_matrices = {**existing_matrices, **self.homography_matrices}
        
        # Write merged matrices back to file
        with open(output_file, 'w') as f:
            json.dump(merged_matrices, f, indent=4)
        
        # Update in-memory matrices with the merged result
        self.homography_matrices = merged_matrices
        
        # Report what happened
        if existing_matrices:
            num_new = len(self.homography_matrices) - len(existing_matrices)
            num_updated = len(set(existing_matrices.keys()) & set(self.homography_matrices.keys()))
            if num_new > 0:
                print(f"Added {num_new} new homography matrices")
            if num_updated > 0:
                print(f"Updated {num_updated} existing homography matrices")
        else:
            print(f"Saved {len(self.homography_matrices)} homography matrices to new file {output_file}")
    
    def load_homography_matrices(self, input_file: str = "homography_matrices.json") -> None:
        """
        Load homography matrices from a JSON file
        
        Args:
            input_file: Path to the JSON file
        """
        if not os.path.exists(input_file):
            print(f"File {input_file} not found")
            return
        
        try:
            with open(input_file, 'r') as f:
                loaded_matrices = json.load(f)
            
            # Merge with any existing matrices (loaded take precedence)
            self.homography_matrices = {**self.homography_matrices, **loaded_matrices}
            
            print(f"Loaded {len(loaded_matrices)} homography matrices from {input_file}")
        except json.JSONDecodeError:
            print(f"Error: File {input_file} contains invalid JSON data")
        except Exception as e:
            print(f"Error loading file {input_file}: {str(e)}")
    
    def visualize_homography(self, room_index: int, cam_index: int) -> None:
        """
        Visualize the homography projection from camera to map
        
        Args:
            room_index: Room index
            cam_index: Camera index
        """
        room = f"room{room_index}"
        cam = f"cam{cam_index}"
        key = f"{room}_{cam}"
        
        if key not in self.homography_matrices:
            print(f"No homography matrix found for {key}")
            return
        
        # Select the room and camera to load images
        self.select_room(room_index)
        self.select_camera(cam_index)
        
        # Get homography matrix
        H = np.array(self.homography_matrices[key]["matrix"], dtype=np.float32)
        
        # Warp camera image to map perspective
        warped_img = cv2.warpPerspective(
            self.cam_image, 
            H, 
            (self.map_image.shape[1], self.map_image.shape[0])
        )
        
        # Create blended visualization
        alpha = 0.5
        blended = cv2.addWeighted(self.map_image, 1-alpha, warped_img, alpha, 0)
        
        # Plot the results
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(self.cam_image)
        plt.title(f"Camera Image ({cam})")
        
        plt.subplot(2, 2, 2)
        plt.imshow(self.map_image)
        plt.title("Map Image")
        
        plt.subplot(2, 2, 3)
        plt.imshow(warped_img)
        plt.title("Warped Camera Image")
        
        plt.subplot(2, 2, 4)
        plt.imshow(blended)
        plt.title("Blended Result")
        
        plt.tight_layout()
        plt.show()
        
    def visualize_all_cameras(self, room_index: int) -> None:
        """
        Visualize homography projections from all cameras in a room blended together
        
        Args:
            room_index: Room index
        """
        room = f"room{room_index}"
        
        # Select the room to load the map image
        try:
            self.select_room(room_index)
        except Exception as e:
            print(f"Error selecting room: {str(e)}")
            return
        
        # Find all cameras for this room in the homography matrices
        room_cameras = [key for key in self.homography_matrices.keys() if key.startswith(room)]
        
        if not room_cameras:
            print(f"No homography matrices found for {room}")
            return
        
        print(f"Found {len(room_cameras)} camera(s) with homography matrices for {room}")
        
        # Create a base map for blending (convert to float for better blending)
        base_map = self.map_image.copy().astype(np.float32) / 255.0
        
        # Create individual warped images for each camera
        warped_images = []
        camera_names = []
        
        # Process each camera
        for key in room_cameras:
            cam_name = key.split('_')[1]  # Extract camera name from key (e.g., "cam0")
            cam_idx = int(cam_name.replace("cam", ""))
            camera_names.append(cam_name)
            
            try:
                # Load the camera image
                self.select_camera(cam_idx)
                
                # Get homography matrix
                H = np.array(self.homography_matrices[key]["matrix"], dtype=np.float32)
                
                # Warp camera image to map perspective - convert to float for better blending
                warped = cv2.warpPerspective(
                    self.cam_image.astype(np.float32) / 255.0, 
                    H, 
                    (self.map_image.shape[1], self.map_image.shape[0])
                )
                
                # Store the warped image
                warped_images.append(warped)
                
            except Exception as e:
                print(f"Error processing {cam_name}: {str(e)}")
        
        # Plot the results
        if warped_images:
            # Create a properly blended image - blend all camera views first
            n_cameras = len(warped_images)
            
            # Start with a black image where camera pixels will be added
            camera_blend = np.zeros_like(base_map)
            
            # Create masks to track where camera pixels have been added
            camera_mask = np.zeros((base_map.shape[0], base_map.shape[1]), dtype=np.float32)
            
            # Add each camera to the blend
            for i, warped in enumerate(warped_images):
                # Create a mask for non-zero (actual camera view) pixels
                current_mask = np.any(warped > 0.05, axis=2).astype(np.float32)
                
                # Add this camera's pixels to the blend
                camera_blend += warped * np.stack([current_mask, current_mask, current_mask], axis=2)
                
                # Add to the mask
                camera_mask += current_mask
            
            # Avoid division by zero
            camera_mask = np.maximum(camera_mask, 1e-5)
            
            # Normalize by the number of cameras that contribute to each pixel
            camera_blend /= np.stack([camera_mask, camera_mask, camera_mask], axis=2)
            
            # Now blend the combined cameras with the map
            alpha_map = 0.3  # Map visibility
            alpha_cams = 0.7  # Cameras visibility
            blended = alpha_map * base_map + alpha_cams * camera_blend
            
            # Ensure pixel values are in valid range
            blended = np.clip(blended, 0, 1)
            
            # Convert back to uint8 for display
            blended_uint8 = (blended * 255).astype(np.uint8)
            
            # Convert individual warped images back to uint8
            warped_uint8 = [(w * 255).astype(np.uint8) for w in warped_images]
            map_uint8 = (base_map * 255).astype(np.uint8)
            
            # Determine grid size based on number of cameras
            grid_size = int(np.ceil(np.sqrt(n_cameras + 2)))  # +2 for map and final blend
            
            plt.figure(figsize=(15, 10))
            
            # Plot map image
            plt.subplot(grid_size, grid_size, 1)
            plt.imshow(map_uint8)
            plt.title("Map Image")
            
            # Plot each warped camera image
            for i, (warped, cam_name) in enumerate(zip(warped_uint8, camera_names)):
                plt.subplot(grid_size, grid_size, i + 2)
                plt.imshow(warped)
                plt.title(f"Warped {cam_name}")
            
            # Plot final blended result
            plt.subplot(grid_size, grid_size, len(warped_uint8) + 2)
            plt.imshow(blended_uint8)
            plt.title("All Cameras Blended")
            
            plt.tight_layout()
            plt.show()
            
            # Also show a full-screen version of the final blend with a color bar
            plt.figure(figsize=(12, 10))
            plt.imshow(blended_uint8)
            plt.title(f"All {len(warped_uint8)} Cameras Blended - {room}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            # Create a visualization with colored camera coverage
            plt.figure(figsize=(12, 10))
            
            # Start with a grayscale map
            gray_map = cv2.cvtColor(self.map_image, cv2.COLOR_RGB2GRAY)
            colored_visualization = np.stack([gray_map, gray_map, gray_map], axis=2)
            
            # Assign different colors to each camera
            colors = [
                [255, 0, 0],   # Red
                [0, 255, 0],   # Green
                [0, 0, 255],   # Blue
                [255, 255, 0], # Yellow
                [255, 0, 255], # Magenta
                [0, 255, 255], # Cyan
                [255, 128, 0], # Orange
                [128, 0, 255]  # Purple
            ]
            
            # Create a colored visualization
            for i, warped in enumerate(warped_images):
                # Get a color for this camera
                color = colors[i % len(colors)]
                
                # Create a mask for non-zero pixels
                mask = np.any(warped > 0.05, axis=2).astype(np.float32)
                
                # Add colored overlay
                for c in range(3):
                    colored_visualization[:, :, c] = np.where(
                        mask > 0,
                        colored_visualization[:, :, c] * 0.5 + color[c] * 0.5 * mask,
                        colored_visualization[:, :, c]
                    )
            
            # Display the colored visualization
            plt.imshow(colored_visualization.astype(np.uint8))
            plt.title(f"Camera Coverage Map - {room}")
            
            # Add a legend
            for i, cam_name in enumerate(camera_names):
                color = colors[i % len(colors)]
                plt.plot([], [], 's', color=[c/255 for c in color], label=cam_name)
            
            plt.legend(loc='upper right')
            plt.axis('off')
            plt.tight_layout()
            plt.show()


def main():
    # Example usage
    tool = HomographyTool()
    
    print("Available rooms:")
    for i, room in enumerate(tool.room_paths):
        print(f"{i}: {room}")
    
    # User selects a room
    room_idx = int(input("Select a room index: "))
    tool.select_room(room_idx)
    
    # Get and display available cameras
    cameras = tool.get_cameras_for_room()
    print("\nAvailable cameras:")
    for i, cam in enumerate(cameras):
        print(f"{i}: {cam}")
    
    while True:
        # User selects a camera
        cam_idx = int(input("\nSelect a camera index (or -1 to finish): "))
        if cam_idx < 0:
            break
            
        cam_name = cameras[cam_idx]
        actual_cam_idx = int(cam_name.replace("cam", ""))
        tool.select_camera(actual_cam_idx)
        
        # User selects points
        print("\nSelect corresponding points on the camera and map images:")
        print("1. First click on the camera image")
        print("2. Then click on the corresponding point on the map image")
        print("3. Repeat until you have enough points (at least 4)")
        print("4. Close the window when done")
        
        tool.select_points()
        
        # Calculate homography
        tool.calculate_homography()
    
    # Save homography matrices
    tool.save_homography_matrices()
    
    # Ask if user wants to visualize
    visualize = input("\nDo you want to visualize the results? (y/n): ")
    if visualize.lower() == 'y':
        room_idx = int(input("Enter room index: "))
        cam_idx = int(input("Enter camera index: "))
        tool.visualize_homography(room_idx, cam_idx)


if __name__ == "__main__":
    main()
