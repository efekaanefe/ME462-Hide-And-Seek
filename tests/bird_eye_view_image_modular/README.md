# Homography Projection Tool

This tool allows you to create homography projections from camera images to a 2D map. It provides functionality to:

1. Select corresponding points between camera views and map images
2. Calculate homography matrices
3. Visualize the projections
4. Save and load homography matrices

## Directory Structure

The tool expects the following directory structure:

```
rooms_database/
    room{index}/
        cam{index}/
            # Camera images
        2Dmap.png  # 2D map image of the room
```

## Requirements

- Python 3.6+
- OpenCV (cv2)
- NumPy
- Matplotlib

## Setup

You can set up the required directory structure using the provided setup script:

```bash
# Create the directory structure
python setup_database.py create --rooms 2 --cameras 3

# Add a map image to a room
python setup_database.py add-map --room 0 --map /path/to/map.png

# Add camera images to a camera
python setup_database.py add-camera --room 0 --camera 0 --images /path/to/image1.jpg /path/to/image2.jpg
```

## Usage

### Basic Usage

Run the main script:

```bash
python homography_modular.py
```

### Demo Script

For easier usage, you can also use the demo script:

```bash
python homography_demo.py
```

The demo script provides options to:
1. Visualize saved homographies
2. Create homography for a new camera
3. Add points to an existing homography

### Workflow

1. Select a room by index
2. For each camera in the room:
   - Select the camera
   - Select at least 4 corresponding points between the camera view and the map
   - Calculate the homography matrix
3. Save all homography matrices
4. Visualize the projections

### Interactive Point Selection

The tool opens a window with two panels:
- Left panel: Camera image
- Right panel: Map image

To select corresponding points:
1. Click on a point in the camera image
2. Click on the corresponding point in the map image
3. Repeat until you have at least 4 point pairs
4. Close the window

### Visualization

The tool can visualize the homography projection with:
- Original camera image
- Map image
- Warped camera image
- Blended result

## API Reference

### `HomographyTool` Class

Main class for the homography tool.

#### Methods

- `select_room(room_index)`: Select a room by index
- `select_camera(cam_index)`: Select a camera by index
- `select_points()`: Open interactive window to select points
- `calculate_homography()`: Calculate homography matrix
- `save_homography_matrices(output_file)`: Save matrices to JSON
- `load_homography_matrices(input_file)`: Load matrices from JSON
- `visualize_homography(room_index, cam_index)`: Visualize projection 