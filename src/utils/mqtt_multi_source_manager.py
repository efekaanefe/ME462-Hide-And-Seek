import paho.mqtt.client as mqtt
import json
import time
import threading
from collections import defaultdict, deque
from collections import defaultdict
from typing import Dict, List, Optional, Callable
import logging


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

