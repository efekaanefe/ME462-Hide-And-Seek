import paho.mqtt.client as mqtt
import time
import random

class MQTTPublisher:
    def __init__(self, broker_address, port=1883, client_id_prefix="basic_pub"):
        self.broker_address = broker_address
        self.port = port
        self.client_id = f"{client_id_prefix}-{random.randint(0, 100000)}"
        self.is_connected = False

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, self.client_id, protocol=mqtt.MQTTv5)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish # Basic acknowledgement

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == mqtt.CONNACK_ACCEPTED:
            self.is_connected = True
            print(f"Publisher [{self.client_id}]: Connected to {self.broker_address}")
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
        # This callback confirms the message acknowledgment from the broker for QoS > 0
        # For a very basic publisher, you might not need to act on this in detail.
        # Reason_code might be an int (0 for success with QoS 1 from v3.1.1 broker) or a ReasonCode object.
        if (isinstance(reason_code, int) and reason_code == 0) or \
           (hasattr(reason_code, 'is_failure') and not reason_code.is_failure):
            print(f"Publisher [{self.client_id}]: Message MID {mid} published.")
        else:
            print(f"Publisher [{self.client_id}]: Message MID {mid} publish failed or acknowledged with issue: {reason_code}")


    def connect(self):
        try:
            print(f"Publisher [{self.client_id}]: Attempting to connect...")
            self.client.connect(self.broker_address, self.port, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            print(f"Publisher [{self.client_id}]: Connection exception: {e}")
            self.is_connected = False

    def publish(self, topic, message, qos=0, retain=False): # Defaulting to QoS 0 for simplicity
        if not self.is_connected:
            print(f"Publisher [{self.client_id}]: Not connected. Cannot publish.")
            return False
        
        result = self.client.publish(topic, payload=message, qos=qos, retain=retain)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            # For QoS 0, on_publish might not be called by some brokers/configs,
            # or called immediately by Paho.
            # For QoS 1/2, _on_publish will be called on PUBACK/PUBCOMP.
            # print(f"Publisher [{self.client_id}]: Message sent to topic '{topic}' (MID: {result.mid}, QoS: {qos})")
            return True
        else:
            print(f"Publisher [{self.client_id}]: Failed to send message. Error: {mqtt.error_string(result.rc)}")
            return False

    def disconnect(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            print(f"Publisher [{self.client_id}]: Disconnecting.")

# --- Example Usage for BasicMQTTPublisher ---
if __name__ == "__main__":
    publisher = MQTTPublisher(broker_address="mqtt.eclipseprojects.io")
    publisher.connect()

    # Wait a bit for connection
    time.sleep(2) 

    if publisher.is_connected:
        print("\n--- Publishing Frame Data (Example) ---")
        for i in range(5): # Simulate 5 frames
            frame_data = f"Frame {i} data: value={random.random():.2f}"
            print(f"Frame {i}: Publishing '{frame_data}'")
            publisher.publish("game/player/position", frame_data, qos=1) # Use QoS 1 to get _on_publish callback
            time.sleep(1) # Simulate time between frames
    else:
        print("Publisher not connected for example run.")

    publisher.disconnect()
    print("Basic Publisher example finished.")
