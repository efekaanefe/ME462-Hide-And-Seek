# Updated Manager Code (mqtt_manager.py)
import paho.mqtt.client as mqtt
import json
import time
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Callable
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict
from matplotlib.patches import Circle

class MQTTMultiSourceManager:
    def __init__(self, broker_address: str, port: int = 1883, client_id_prefix: str = "multi_manager"):
        self.broker_address = broker_address
        self.port = port
        self.client_id = f"{client_id_prefix}-{int(time.time())}"
        self.is_connected = False
        
        # Data storage
        self.tracks_data = defaultdict(dict)  # {source: {track_id: track_data}}
        self.message_history = defaultdict(lambda: deque(maxlen=100))  # Keep last 100 messages per source
        self.lock = threading.Lock()
        
        # Source configuration
        self.sources = {}  # {source_name: topic_pattern}
        self.message_handlers = {}  # {source_name: handler_function}
        
        # Callbacks
        self.on_track_update_callback = None
        self.on_new_track_callback = None
        self.on_track_lost_callback = None
        
        # MQTT Client setup
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, self.client_id, protocol=mqtt.MQTTv5)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
        # Track timeout handling
        self.track_timeout = 1.0  # seconds
        self.last_seen = defaultdict(dict)  # {source: {track_id: timestamp}}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"MQTTManager-{self.client_id}")

    def add_source(self, source_name: str, topic_pattern: str, message_handler: Optional[Callable] = None):
        """
        Add a data source to monitor
        
        Args:
            source_name: Identifier for the source (e.g., "room1_camera1")
            topic_pattern: MQTT topic pattern to subscribe to (e.g., "tracking/room1/camera1/position/+")
            message_handler: Optional custom handler function for this source
        """
        self.sources[source_name] = topic_pattern
        if message_handler:
            self.message_handlers[source_name] = message_handler
        self.logger.info(f"Added source '{source_name}' with topic pattern '{topic_pattern}'")

    def set_callbacks(self, on_track_update=None, on_new_track=None, on_track_lost=None):
        """Set callback functions for track events"""
        self.on_track_update_callback = on_track_update
        self.on_new_track_callback = on_new_track
        self.on_track_lost_callback = on_track_lost

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == mqtt.CONNACK_ACCEPTED:
            self.is_connected = True
            self.logger.info(f"Connected to {self.broker_address}")
            
            # Subscribe to all configured sources
            for source_name, topic_pattern in self.sources.items():
                result, mid = self.client.subscribe(topic_pattern, qos=1)
                if result == mqtt.MQTT_ERR_SUCCESS:
                    self.logger.info(f"Subscribed to {topic_pattern} for source '{source_name}'")
                else:
                    self.logger.error(f"Failed to subscribe to {topic_pattern}: {mqtt.error_string(result)}")
        else:
            self.is_connected = False
            self.logger.error(f"Failed to connect. Reason: {reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self.is_connected = False
        if reason_code == mqtt.MQTT_ERR_SUCCESS or reason_code is None:
            self.logger.info("Disconnected")
        else:
            self.logger.warning(f"Unexpectedly disconnected. Reason: {reason_code}")

    def _on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages"""
        try:
            # Decode payload
            payload = msg.payload.decode('utf-8')
            data = json.loads(payload)
            
            # Determine source from topic
            source_name = self._get_source_from_topic(msg.topic)
            if not source_name:
                self.logger.warning(f"Unknown source for topic: {msg.topic}")
                return
            
            # Process the message
            self._process_message(source_name, msg.topic, data)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON from {msg.topic}: {e}")
        except Exception as e:
            self.logger.error(f"Error processing message from {msg.topic}: {e}")

    def _get_source_from_topic(self, topic: str) -> Optional[str]:
        """Determine which source a topic belongs to"""
        for source_name, topic_pattern in self.sources.items():
            if self._topic_matches_pattern(topic, topic_pattern):
                return source_name
        return None

    def _topic_matches_pattern(self, topic: str, pattern: str) -> bool:
        """MQTT topic pattern matching with + and # wildcards"""
        topic_parts = topic.split('/')
        pattern_parts = pattern.split('/')
        
        # Handle # wildcard (matches everything after this point)
        if '#' in pattern_parts:
            hash_index = pattern_parts.index('#')
            if hash_index != len(pattern_parts) - 1:
                return False  # # must be last
            pattern_parts = pattern_parts[:hash_index]
            topic_parts = topic_parts[:hash_index]
        
        # Must have same number of parts after handling #
        if len(topic_parts) != len(pattern_parts):
            return False
        
        # Check each part
        for topic_part, pattern_part in zip(topic_parts, pattern_parts):
            if pattern_part != '+' and pattern_part != topic_part:
                return False
        
        return True

    def _process_message(self, source_name: str, topic: str, data: dict):
        """Process incoming message data"""
        with self.lock:
            # Extract track information
            #print(type(data)) # dictionary
            
            track_id = data.get('track_id')
            if not track_id:
                self.logger.warning(f"Message from {source_name} missing track_id")
                return
            
            # Check if this is a new track
            is_new_track = track_id not in self.tracks_data[source_name]
            
            # Update track data
            previous_data = self.tracks_data[source_name].get(track_id, {})
            self.tracks_data[source_name][track_id] = {
                'track_id': track_id,
                'name': data.get('name', 'Unknown'),
                'x': float(data.get('x', 0)),
                'y': float(data.get('y', 0)),
                'orientation': float(data.get('orientation', 0)) if data.get('orientation') is not None else None,
                'timestamp': data.get('timestamp', time.time()),
                'source': source_name,
                'topic': topic,
                'room_index': data.get('room_index'),
                'camera_index': data.get('camera_index'),
                'last_updated': time.time()
            }
            
            # Update last seen timestamp
            self.last_seen[source_name][track_id] = time.time()
            
            # Add to message history
            self.message_history[source_name].append({
                'timestamp': time.time(),
                'track_id': track_id,
                'data': data
            })
            
            # Custom message handler for this source
            if source_name in self.message_handlers:
                try:
                    self.message_handlers[source_name](source_name, track_id, data, previous_data)
                except Exception as e:
                    self.logger.error(f"Error in custom handler for {source_name}: {e}")
            
            # Trigger callbacks
            if is_new_track and self.on_new_track_callback:
                try:
                    self.on_new_track_callback(source_name, track_id, self.tracks_data[source_name][track_id])
                except Exception as e:
                    self.logger.error(f"Error in new track callback: {e}")
            elif not is_new_track and self.on_track_update_callback:
                try:
                    self.on_track_update_callback(source_name, track_id, self.tracks_data[source_name][track_id], previous_data)
                except Exception as e:
                    self.logger.error(f"Error in track update callback: {e}")

    def connect(self):
        """Connect to MQTT broker"""
        try:
            self.logger.info("Attempting to connect...")
            self.client.connect(self.broker_address, self.port, keepalive=60)
            self.client.loop_start()
            
            # Start cleanup thread
            cleanup_thread = threading.Thread(target=self._cleanup_old_tracks, daemon=True)
            cleanup_thread.start()
            
        except Exception as e:
            self.logger.error(f"Connection exception: {e}")
            self.is_connected = False

    def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.logger.info("Disconnecting")

    def _cleanup_old_tracks(self):
        """Remove tracks that haven't been updated recently"""
        while True:
            try:
                current_time = time.time()
                tracks_to_remove = []
                
                with self.lock:
                    for source_name in list(self.last_seen.keys()):
                        for track_id in list(self.last_seen[source_name].keys()):
                            if current_time - self.last_seen[source_name][track_id] > self.track_timeout:
                                tracks_to_remove.append((source_name, track_id))
                
                # Remove old tracks
                for source_name, track_id in tracks_to_remove:
                    with self.lock:
                        if track_id in self.tracks_data[source_name]:
                            track_data = self.tracks_data[source_name].pop(track_id)
                            del self.last_seen[source_name][track_id]
                            
                            # Trigger callback
                            if self.on_track_lost_callback:
                                try:
                                    self.on_track_lost_callback(source_name, track_id, track_data)
                                except Exception as e:
                                    self.logger.error(f"Error in track lost callback: {e}")
                            
                            self.logger.info(f"Removed old track {track_id} from {source_name}")
                
                time.sleep(0.1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in cleanup thread: {e}")
                time.sleep(0.5)

    # Data access methods
    def get_all_tracks(self) -> Dict[str, Dict[str, dict]]:
        """Get all current tracks from all sources"""
        with self.lock:
            return dict(self.tracks_data)

    def get_tracks_by_source(self, source_name: str) -> Dict[str, dict]:
        """Get tracks from a specific source"""
        with self.lock:
            return dict(self.tracks_data.get(source_name, {}))

    def get_track(self, source_name: str, track_id: str) -> Optional[dict]:
        """Get a specific track from a source"""
        with self.lock:
            return self.tracks_data.get(source_name, {}).get(track_id)

    def get_all_tracks_combined(self) -> List[dict]:
        """Get all tracks from all sources as a flat list"""
        with self.lock:
            all_tracks = []
            for source_tracks in self.tracks_data.values():
                all_tracks.extend(source_tracks.values())
            return all_tracks

    def get_message_history(self, source_name: str, limit: int = 10) -> List[dict]:
        """Get recent message history for a source"""
        with self.lock:
            history = list(self.message_history[source_name])
            return history[-limit:] if limit else history

    def get_stats(self) -> dict:
        """Get statistics about tracked data"""
        with self.lock:
            stats = {
                'sources': list(self.sources.keys()),
                'total_tracks': sum(len(tracks) for tracks in self.tracks_data.values()),
                'tracks_per_source': {source: len(tracks) for source, tracks in self.tracks_data.items()},
                'is_connected': self.is_connected,
                'uptime': time.time() - (self.client._sock_connect_time if hasattr(self.client, '_sock_connect_time') else time.time())
            }
            return stats


# Camera-specific handlers
def camera_handler(source_name: str, track_id: str, data: dict, previous_data: dict):
    """Generic camera handler"""
    room = data.get('room_index', 'Unknown')
    camera = data.get('camera_index', 'Unknown')
    name = data.get('name', 'Unknown')
    print(f"{source_name.upper()} (Room {room}, Cam {camera}): Track {track_id} ({name}) at ({data['x']:.2f}, {data['y']:.2f})")


# Enhanced callback functions
def on_new_track(source_name: str, track_id: str, track_data: dict):
    room = track_data.get('room_index', 'Unknown')
    camera = track_data.get('camera_index', 'Unknown')
    name = track_data.get('name', 'Unknown')
    print(f"NEW TRACK: {source_name} (Room {room}, Camera {camera}) detected '{name}' [{track_id}] at ({track_data['x']:.2f}, {track_data['y']:.2f})")

def on_track_update(source_name: str, track_id: str, track_data: dict, previous_data: dict):
    dx = track_data['x'] - previous_data.get('x', 0)
    dy = track_data['y'] - previous_data.get('y', 0)
    if abs(dx) > 0.1 or abs(dy) > 0.1:  # Only log significant movements
        room = track_data.get('room_index', 'Unknown')
        camera = track_data.get('camera_index', 'Unknown')
        print(f"MOVEMENT: {source_name} (Room {room}, Camera {camera}) track {track_id} moved by ({dx:.2f}, {dy:.2f})")

def on_track_lost(source_name: str, track_id: str, track_data: dict):
    room = track_data.get('room_index', 'Unknown')
    camera = track_data.get('camera_index', 'Unknown')
    name = track_data.get('name', 'Unknown')
    print(f"LOST TRACK: {source_name} (Room {room}, Camera {camera}) lost '{name}' [{track_id}] (last seen at {track_data['x']:.2f}, {track_data['y']:.2f})")

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
                (x, y, timestamp) for x, y, timestamp in self.track_positions[name]
                if timestamp >= cutoff_time
            ]
            
            # Remove tracks with no recent positions
            if not self.track_positions[name]:
                del self.track_positions[name]
    
    def update_track_position(self, name, x, y, timestamp=None):
        """Add a new position for a named track"""
        if timestamp is None:
            timestamp = time.time()
            
        self.track_positions[name].append((x, y, timestamp))
        
        # Clean up old data
        self.cleanup_old_tracks()
        
        print(f"Added position for {name}: ({x:.2f}, {y:.2f}) - Recent positions: {len(self.track_positions[name])}")
    
    def get_average_positions(self):
        """Calculate average positions for each named track (last n seconds only)"""
        # Clean up old data first
        self.cleanup_old_tracks()
        
        averages = {}
        for name, positions in self.track_positions.items():
            if positions:
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                timestamps = [pos[2] for pos in positions]
                
                averages[name] = {
                    'x': np.mean(x_coords),
                    'y': np.mean(y_coords),
                    'count': len(positions),
                    'time_span': max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
                }
        return averages
    
    def update_visualization(self):
        """Update the map visualization with current average positions"""
        if self.ax is None:
            return
            
        # Clear previous plot
        self.ax.clear()
        
        # Draw map background
        if self.map_image is not None:
            self.ax.imshow(self.map_image, extent=[
                self.coordinate_bounds['x_min'], self.coordinate_bounds['x_max'],
                self.coordinate_bounds['y_min'], self.coordinate_bounds['y_max']
            ])
        else:
            self.ax.set_xlim(self.coordinate_bounds['x_min'], self.coordinate_bounds['x_max'])
            self.ax.set_ylim(self.coordinate_bounds['y_min'], self.coordinate_bounds['y_max'])
            self.ax.grid(True)
        
        # Plot average positions
        averages = self.get_average_positions()
        colors = plt.cm.tab10(np.linspace(0, 1, len(averages)))
        
        for i, (name, pos_data) in enumerate(averages.items()):
            x, y = pos_data['x'], pos_data['y']
            count = pos_data['count']
            time_span = pos_data['time_span']
            
            # Plot the average position
            color = colors[i % len(colors)]
            self.ax.scatter(x, y, c=[color], s=150, alpha=0.8, 
                          edgecolors='black', linewidth=2)
            
            # Add name label with count and time info
            self.ax.annotate(f'{name} (n={count}, {time_span:.1f}s)', (x, y), 
                           xytext=(10, 10), textcoords='offset points', 
                           fontsize=12, fontweight='bold',
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


# Modified callback functions that work with the original integration
def enhanced_on_track_update(track_data, mapper):
    """Enhanced callback that updates the mapper"""
    name = track_data.get('name', 'Unknown')
    x = track_data.get('x', 0)
    y = track_data.get('y', 0)
    timestamp = track_data.get('timestamp', time.time())
    
    # Update mapper with new position including timestamp
    mapper.update_track_position(name, x, y, timestamp)
    
    # Original callback behavior
    print(f"Track updated: {name} at ({x:.2f}, {y:.2f})")

def enhanced_on_new_track(track_data, mapper):
    """Enhanced callback for new tracks"""
    name = track_data.get('name', 'Unknown')
    x = track_data.get('x', 0)
    y = track_data.get('y', 0)
    timestamp = track_data.get('timestamp', time.time())
    
    # Update mapper
    mapper.update_track_position(name, x, y, timestamp)
    
    # Original callback behavior  
    print(f"New track: {name} [{track_data.get('track_id', '?')}]")

def enhanced_on_track_lost(track_data, mapper):
    """Enhanced callback for lost tracks"""
    name = track_data.get('name', 'Unknown')
    print(f"Track lost: {name} [{track_data.get('track_id', '?')}]")

# Usage example in main:
if __name__ == "__main__":
    # Initialize the track mapper with 30-second time window
    MAP_IMAGE_PATH = "room_database//2Dmap.png"
    COORDINATE_BOUNDS = {
        'x_min': 0, 'x_max': 1000,
        'y_min': 0, 'y_max': 1000
    }
    TIME_WINDOW_SECONDS = 2  # Only keep last 30 seconds of data
    
    # Create the mapper
    mapper = TrackMapper(MAP_IMAGE_PATH, COORDINATE_BOUNDS, TIME_WINDOW_SECONDS)
    
    # Create MQTT manager
    manager = MQTTMultiSourceManager(broker_address="mqtt.eclipseprojects.io")
    manager.add_source("all_tracking", "tracking/+/+/+", camera_handler)
    
    # Set enhanced callbacks that include the mapper
    manager.set_callbacks(
        on_track_update=lambda source, track_id, track_data, prev: enhanced_on_track_update(track_data, mapper),
        on_new_track=lambda source, track_id, track_data: enhanced_on_new_track(track_data, mapper),
        on_track_lost=lambda source, track_id, track_data: enhanced_on_track_lost(track_data, mapper)
    )
    
    manager.connect()
    
    try:
        iteration = 0
        while True:
            if manager.is_connected:
                iteration += 1
                
                # Update visualization every iteration
                if iteration % 1 == 0:
                    mapper.update_visualization()
                    mapper.print_summary()
                
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        manager.disconnect()
        plt.close('all')
        print("Mapper stopped.")