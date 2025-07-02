import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict



try:
    import socket
    PORT = 9999
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("192.168.68.73", PORT))
except Exception:
    print("Cant connect to NAO")


class TrackMapper:
    def __init__(self, map_image_path, coordinate_bounds=None, time_window_seconds=30):
        """
        Initialize the track mapper
        
        Args:
            map_image_path: Path to the map image
            coordinate_bounds: Dict with keys 'x_min', 'x_max', 'y_min', 'y_max' 
            time_window_seconds: How many seconds of data to keep for averaging
        """
        self.map_image_path = map_image_path
        self.map_image = None
        self.coordinate_bounds = coordinate_bounds or {
            'x_min': 0, 'x_max': 100, 
            'y_min': 0, 'y_max': 100
        }
        self.time_window_seconds = time_window_seconds
        
        # Store all positions for each name with timestamps: name -> list of (x, y, timestamp)
        self.track_positions = defaultdict(list)
        self.fig = None
        self.ax = None
        self.setup_plot()
        
    def setup_plot(self):
        """Setup the matplotlib plot with the map image"""
        try:
            self.map_image = mpimg.imread(self.map_image_path)
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            plt.ion()  # Turn on interactive mode
        except Exception as e:
            print(f"Error loading map image: {e}")
            # Create a blank plot if image fails to load
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            plt.ion()
    
    def cleanup_old_tracks(self):
        """Remove position data older than time_window_seconds"""
        current_time = time.time()
        cutoff_time = current_time - self.time_window_seconds
        
        for name in list(self.track_positions.keys()):
            # Filter out old positions
            self.track_positions[name] = [
                pos for pos in self.track_positions[name]
                if pos[2] >= cutoff_time  # timestamp is at index 2
            ]
            
            # Remove tracks with no recent positions
            if not self.track_positions[name]:
                del self.track_positions[name]

    def update_track_position(self, name, x, y, timestamp=None, orientation=None):
        """Add a new position for a named track with optional orientation"""
        if timestamp is None:
            timestamp = time.time()
        
        # Store as (x, y, timestamp, orientation) - orientation can be None
        self.track_positions[name].append((x, y, timestamp, orientation))
        
        # Clean up old data
        self.cleanup_old_tracks()    


    def get_average_positions(self):
        """Calculate average positions and orientations for each named track (last n seconds only)"""
        # Clean up old data first
        self.cleanup_old_tracks()
        averages = {}
        for name, positions in self.track_positions.items():
            if positions:
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                timestamps = [pos[2] for pos in positions]
                orientations = [pos[3] for pos in positions if len(pos) > 3 and pos[3] is not None]  # Get orientations if available
                
                avg_data = {
                    'x': np.mean(x_coords),
                    'y': np.mean(y_coords),
                    'count': len(positions),
                    'time_span': max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
                }
                
                # Add average orientation if available
                if orientations:
                    avg_data['orientation'] = np.mean(orientations)
                
                averages[name] = avg_data
        return averages

    def update_visualization(self):
        """Update the map visualization with current average positions and orientations"""
        if self.ax is None:
            return
        
        # Clear previous plot
        self.ax.clear()
        
        # Set constant limits first
        self.ax.set_xlim(self.coordinate_bounds['x_min'], self.coordinate_bounds['x_max'])
        self.ax.set_ylim(self.coordinate_bounds['y_min'], self.coordinate_bounds['y_max'])
        
        # Draw map background if available
        if self.map_image is not None:
            self.ax.imshow(self.map_image, extent=[
                self.coordinate_bounds['x_min'], self.coordinate_bounds['x_max'],
                self.coordinate_bounds['y_min'], self.coordinate_bounds['y_max']
            ])
        else:
            self.ax.grid(True)
        
        # Plot average positions
        averages = self.get_average_positions()
        colors = plt.cm.tab10(np.linspace(0, 1, len(averages)))
        
        for i, (name, pos_data) in enumerate(averages.items()):
            x, y = pos_data['x'], pos_data['y']
            y = 1000 - y
            count = pos_data['count']
            time_span = pos_data['time_span']
            
            # Plot the average position
            color = colors[i % len(colors)]
            self.ax.scatter(x, y, c=[color], s=150, alpha=0.8,
                        edgecolors='black', linewidth=2)
            
            # Plot orientation arrow if available
            if 'orientation' in pos_data:
                orientation = pos_data['orientation']
                # Arrow length proportional to plot size
                arrow_length = min(self.coordinate_bounds['x_max'] - self.coordinate_bounds['x_min'],
                                self.coordinate_bounds['y_max'] - self.coordinate_bounds['y_min']) * 0.05
                dx = arrow_length * np.cos(orientation)
                dy = -arrow_length * np.sin(orientation)
                
                self.ax.arrow(x, y, dx, dy, head_width=arrow_length*0.3, 
                            head_length=arrow_length*0.2, fc=color, ec=color, alpha=0.8)
            
            # Add name label with count and time info
            self.ax.annotate(f'{name} (n={count}, {time_span:.1f}s)',
                            (x, y), xytext=(10, 10),
                            textcoords='offset points', fontsize=12,
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.5',
                                    facecolor='white', alpha=0.8))
        
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_title(f'Average Track Positions - Last {self.time_window_seconds}s ({len(averages)} active tracks)')
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def print_summary(self):
        """Print summary of all tracked positions"""
        print(f"\n--- TRACK POSITION SUMMARY (Last {self.time_window_seconds}s) ---")
        averages = self.get_average_positions()
        for name, pos_data in averages.items():
            print(f"{name}: Average position ({pos_data['x']:.2f}, {pos_data['y']:.2f}) "
                  f"from {pos_data['count']} observations over {pos_data['time_span']:.1f}s")
            
    def handle_nao_angle(self, target="Target"):
        """Compute relative angle from NAO to target using latest raw position data."""
        def angle_diff(a, b):
            """Calculate smallest difference between two angles in degrees."""
            d = a - b
            return ((d + 180) % 360) - 180

        nao_found = False
        target_found = False

        if "NAO" in self.track_positions and self.track_positions["NAO"]:
            last_nao = self.track_positions["NAO"][-1]
            x_nao, y_nao, _, orientation_nao = last_nao
            orientation_nao = np.degrees(orientation_nao)  # convert to degrees
            nao_found = True
            print("NAO found")
        else:
            print("Error: NAO not found")

        if target in self.track_positions and self.track_positions[target]:
            last_target = self.track_positions[target][-1]
            x_target, y_target, _, _ = last_target
            target_found = True
            print("Target found")
        else:
            print(f"Error: Target '{target}' not found")

        if not (nao_found and target_found):
            return None

        # Calculate direction to target
        dx = x_target - x_nao
        dy = y_target - y_nao
        angle_to_target = np.degrees(np.arctan2(dy, dx))

        # Calculate relative angle
        relative_angle = angle_diff(angle_to_target, orientation_nao)

        print(f"Angle to target: {angle_to_target:.2f}")
        print(f"NAO orientation: {orientation_nao:.2f}")
        print(f"Relative angle: {relative_angle:.2f}")

        self.send_to_nao(np.radians(relative_angle))  # send in radians

        


    def send_to_nao(self, angle_deg):
        # Check if angle is valid before sending
        if angle_deg is None:
            print("Cannot send invalid angle to NAO")
            return False
            
        # Try to send data over socket
        try:
            message = f"({angle_deg:.2f},{angle_deg:.2f})\n"
            # Uncomment to send over socket
            sock.sendall(message.encode())
            print(f"Sending to NAO: {message.strip()}")
            return True
        except (BrokenPipeError, ConnectionResetError):
            print("Connection closed.")
            return False
        except Exception as e:
            print(f"Error sending to NAO: {e}")
            return False

        
        
# Modified callback functions that work with orientation data
def enhanced_on_track_update(track_data, mapper):
    """Enhanced callback that updates the mapper with orientation"""
    name = track_data.get('name', 'Unknown')
    x = track_data.get('x', 0)
    y = track_data.get('y', 0)
    timestamp = track_data.get('timestamp', time.time())
    orientation = track_data.get('orientation')  # This can be None
    
    # Update mapper with new position including orientation
    mapper.update_track_position(name, x, y, timestamp, orientation)
    
    # Original callback behavior
    if orientation is not None:
        print(f"Track updated: {name} at ({x:.2f}, {y:.2f}) facing {np.degrees(orientation):.1f}°")
    else:
        print(f"Track updated: {name} at ({x:.2f}, {y:.2f})")

def enhanced_on_new_track(track_data, mapper):
    """Enhanced callback for new tracks with orientation"""
    name = track_data.get('name', 'Unknown')
    x = track_data.get('x', 0)
    y = track_data.get('y', 0)
    timestamp = track_data.get('timestamp', time.time())
    orientation = track_data.get('orientation')
    
    # Update mapper
    mapper.update_track_position(name, x, y, timestamp, orientation)
    
    # Original callback behavior  
    if orientation is not None:
        print(f"New track: {name} [{track_data.get('track_id', '?')}] at ({x:.2f}, {y:.2f}) facing {orientation:.1f}°")
    else:
        print(f"New track: {name} [{track_data.get('track_id', '?')}] at ({x:.2f}, {y:.2f})")

def enhanced_on_track_lost(track_data, mapper):
    """Enhanced callback for lost tracks"""
    name = track_data.get('name', 'Unknown')
    print(f"Track lost: {name} [{track_data.get('track_id', '?')}]")