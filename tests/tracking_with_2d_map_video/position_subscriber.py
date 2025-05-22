import time
import json
from MQTTSubscriber import MQTTSubscriber

def main():
    # Initialize MQTT subscriber
    subscriber = MQTTSubscriber(broker_address="mqtt.eclipseprojects.io")
    
    # Subscribe to all player position topics
    subscriber.connect(topics_to_subscribe=["game/player/position/#"])

    print("\n--- Position Subscriber Started ---")
    print("Waiting for position updates...")
    
    try:
        while True:
            if not subscriber.is_connected:
                print("Subscriber trying to reconnect...")
                time.sleep(1)
                continue

            messages = subscriber.get_messages()
            if messages:
                for topic, payload in messages:
                    try:
                        # Parse the position data
                        position_data = eval(payload)  # Convert string representation to dict
                        
                        # Extract data
                        track_id = position_data["track_id"]
                        name = position_data["name"]
                        x = position_data["x"]
                        y = position_data["y"]
                        timestamp = position_data["timestamp"]
                        
                        # Print formatted position data
                        print(f"\nPlayer Update:")
                        print(f"ID: {track_id}")
                        print(f"Name: {name}")
                        print(f"Position: ({x:.2f}, {y:.2f})")
                        print(f"Time: {time.strftime('%H:%M:%S', time.localtime(timestamp))}")
                        print("-" * 30)
                        
                    except Exception as e:
                        print(f"Error processing message: {e}")
                        print(f"Raw payload: {payload}")
            
            time.sleep(0.1)  # Small delay to prevent CPU overuse
            
    except KeyboardInterrupt:
        print("\nSubscriber stopped by user.")
    finally:
        subscriber.disconnect()
        print("Position subscriber finished.")

if __name__ == "__main__":
    main() 