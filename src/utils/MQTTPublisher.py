import paho.mqtt.client as mqtt
import time
import random
import json

class MQTTPublisher:
    def __init__(self, broker_address, port=1883, room_index=None, camera_index=None, client_id_prefix="pub"):
        self.broker_address = broker_address
        self.port = port
        self.room_index = room_index
        self.camera_index = camera_index
        
        # Create a meaningful client ID
        self.client_id = self._generate_client_id(client_id_prefix)
        self.is_connected = False

        # Create base topic structure
        self.base_topic = self._generate_base_topic()
        
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, self.client_id, protocol=mqtt.MQTTv5)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish

    def _generate_client_id(self, prefix):
        """Generate a descriptive client ID based on room and camera info"""
        parts = [prefix]
        
        if self.room_index is not None:
            parts.append(f"room{self.room_index}")
        if self.camera_index is not None:
            parts.append(f"cam{self.camera_index}")
            
        # Add random suffix to ensure uniqueness
        parts.append(str(random.randint(1000, 9999)))
        
        return "-".join(parts)

    def _generate_base_topic(self):
        """Generate base topic structure based on room and camera info"""
        topic_parts = ["tracking"]
        
        if self.room_index is not None:
            topic_parts.append(f"room{self.room_index}")
        if self.camera_index is not None:
            topic_parts.append(f"camera{self.camera_index}")
            
        return "/".join(topic_parts)

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == mqtt.CONNACK_ACCEPTED:
            self.is_connected = True
            print(f"Publisher [{self.client_id}]: Connected to {self.broker_address}")
            print(f"Publisher [{self.client_id}]: Base topic: {self.base_topic}")
        else:
            self.is_connected = False
            print(f"Publisher [{self.client_id}]: Failed to connect. Reason: {reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self.is_connected = False
        if reason_code == mqtt.MQTT_ERR_SUCCESS or reason_code is None:
            print(f"Publisher [{self.client_id}]: Disconnected.")
        else:
            print(f"Publisher [{self.client_id}]: Unexpectedly disconnected. Reason: {reason_code}")

    def _on_publish(self, client, userdata, mid, reason_code, properties):
        if (isinstance(reason_code, int) and reason_code == 0) or \
           (hasattr(reason_code, 'is_failure') and not reason_code.is_failure):
            print(f"Publisher [{self.client_id}]: Message MID {mid} published.")
        else:
            print(f"Publisher [{self.client_id}]: Message MID {mid} publish failed: {reason_code}")

    def connect(self):
        try:
            print(f"Publisher [{self.client_id}]: Attempting to connect...")
            self.client.connect(self.broker_address, self.port, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            print(f"Publisher [{self.client_id}]: Connection exception: {e}")
            self.is_connected = False

    def publish_track_data(self, track_id, track_data, qos=1, retain=False):
        """
        Publish track data with automatic topic generation
        
        Args:
            track_id: Unique identifier for the track
            track_data: Dictionary containing track information (x, y, orientation, etc.)
            qos: Quality of Service level
            retain: Whether to retain the message
        """
        if not self.is_connected:
            print(f"Publisher [{self.client_id}]: Not connected. Cannot publish.")
            return False
        
        # Create full topic
        topic = f"{self.base_topic}/position/{track_id}"
        
        # Ensure track_data has required fields and add metadata
        enhanced_data = {
            "track_id": track_id,
            "room_index": self.room_index,
            "camera_index": self.camera_index,
            "timestamp": time.time(),
            **track_data  # Merge in the provided track data
        }
        
        # Convert to JSON
        message = json.dumps(enhanced_data)
        
        result = self.client.publish(topic, payload=message, qos=qos, retain=retain)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            return True
        else:
            print(f"Publisher [{self.client_id}]: Failed to send message. Error: {mqtt.error_string(result.rc)}")
            return False

    def publish(self, topic, message, qos=0, retain=False):
        """
        Generic publish method (maintains backward compatibility)
        """
        if not self.is_connected:
            print(f"Publisher [{self.client_id}]: Not connected. Cannot publish.")
            return False
        
        result = self.client.publish(topic, payload=message, qos=qos, retain=retain)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            return True
        else:
            print(f"Publisher [{self.client_id}]: Failed to send message. Error: {mqtt.error_string(result.rc)}")
            return False

    def publish_status(self, status_message, qos=1):
        """Publish status information"""
        if not self.is_connected:
            return False
            
        status_topic = f"{self.base_topic}/status"
        status_data = {
            "room_index": self.room_index,
            "camera_index": self.camera_index,
            "client_id": self.client_id,
            "status": status_message,
            "timestamp": time.time()
        }
        
        message = json.dumps(status_data)
        return self.publish(status_topic, message, qos=qos)

    def disconnect(self):
        if self.client:
            # Send offline status before disconnecting
            self.publish_status("offline", qos=1)
            time.sleep(0.1)  # Give it a moment to send
            
            self.client.loop_stop()
            self.client.disconnect()
            print(f"Publisher [{self.client_id}]: Disconnecting.")

    def get_info(self):
        """Get publisher information"""
        return {
            "client_id": self.client_id,
            "room_index": self.room_index,
            "camera_index": self.camera_index,
            "base_topic": self.base_topic,
            "is_connected": self.is_connected,
            "broker_address": self.broker_address
        }


# Example usage patterns:

def example_usage_1():
    """Example 1: Simple room and camera setup"""
    print("=== Example 1: Simple Setup ===")
    
    # Create publisher for Room 1, Camera 2
    publisher = MQTTPublisher(
        broker_address="mqtt.eclipseprojects.io",
        room_index=1,
        camera_index=2
    )
    
    publisher.connect()
    time.sleep(2)
    
    if publisher.is_connected:
        print("Publisher info:", publisher.get_info())
        
        # Send status
        publisher.publish_status("online")
        
        # Example track data
        track_data = {
            "name": "Person_A",
            "x": 123.45,
            "y": 67.89,
            "orientation": 90.0
        }
        
        # Publish track data
        publisher.publish_track_data("track_001", track_data)
        
        time.sleep(1)
    
    publisher.disconnect()

def example_usage_2():
    """Example 2: Multiple publishers for different rooms/cameras"""
    print("\n=== Example 2: Multiple Publishers ===")
    
    publishers = []
    
    # Create publishers for different room/camera combinations
    configs = [
        {"room_index": 1, "camera_index": 1},
        {"room_index": 1, "camera_index": 2},
        {"room_index": 1, "camera_index": 3},
    ]
    
    for config in configs:
        pub = MQTTPublisher(
            broker_address="mqtt.eclipseprojects.io",
            **config
        )
        pub.connect()
        publishers.append(pub)
    
    time.sleep(2)
    
    # Simulate publishing from different sources
    for i, publisher in enumerate(publishers):
        if publisher.is_connected:
            print(f"Publisher {i+1} info:", publisher.get_info())
            
            # Send status
            publisher.publish_status("online")
            
            # Send sample track data
            track_data = {
                "name": f"Track_from_pub_{i+1}",
                "x": random.uniform(0, 100),
                "y": random.uniform(0, 100),
                "orientation": random.uniform(0, 360)
            }
            
            publisher.publish_track_data(f"track_{i+1}_001", track_data)
    
    time.sleep(2)
    
    # Cleanup
    for publisher in publishers:
        publisher.disconnect()



if __name__ == "__main__":
    # Run examples
    #example_usage_1()
    example_usage_2() 
    
    print("\nAll examples completed!")