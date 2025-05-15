import paho.mqtt.client as mqtt
import time
import random
from collections import deque # For efficient message queue

class BasicMQTTSubscriber:
    def __init__(self, broker_address, port=1883, client_id_prefix="basic_sub"):
        self.broker_address = broker_address
        self.port = port
        self.client_id = f"{client_id_prefix}-{random.randint(0, 100000)}"
        self.is_connected = False
        self._messages = deque() # Thread-safe for append/popleft
        self._subscribed_topics_on_connect = [] # List of (topic, qos)
        self._subscribed_topics_map = {} # For on_subscribe context

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, self.client_id, protocol=mqtt.MQTTv5)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.on_subscribe = self._on_subscribe

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == mqtt.CONNACK_ACCEPTED:
            self.is_connected = True
            print(f"Subscriber [{self.client_id}]: Connected to {self.broker_address}")
            if self._subscribed_topics_on_connect:
                self._do_subscribe(self._subscribed_topics_on_connect)
        else:
            self.is_connected = False
            print(f"Subscriber [{self.client_id}]: Failed to connect. Reason: {reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self.is_connected = False
        if reason_code == mqtt.MQTT_ERR_SUCCESS or reason_code is None:
            print(f"Subscriber [{self.client_id}]: Disconnected.")
        else:
            print(f"Subscriber [{self.client_id}]: Unexpectedly disconnected. Reason: {reason_code}")

    def _on_message(self, client, userdata, msg):
        # Store message as a tuple (topic, payload_string)
        try:
            self._messages.append((msg.topic, msg.payload.decode()))
        except UnicodeDecodeError:
            self._messages.append((msg.topic, msg.payload)) # Store as bytes if not decodable
        # print(f"Subscriber [{self.client_id}]: Message received on '{msg.topic}'") # Optional: verbose

    def _on_subscribe(self, client, userdata, mid, reason_codes, properties):
        subscribed_topics_for_mid = self._subscribed_topics_map.get(mid, [])
        for i, rc in enumerate(reason_codes):
            topic = subscribed_topics_for_mid[i] if i < len(subscribed_topics_for_mid) else "Unknown Topic"
            if not rc.is_failure:
                print(f"Subscriber [{self.client_id}]: Subscribed to '{topic}' (QoS: {rc.value})")
            else:
                print(f"Subscriber [{self.client_id}]: Failed to subscribe to '{topic}'. Reason: {rc}")
        if mid in self._subscribed_topics_map:
            del self._subscribed_topics_map[mid]

    def _do_subscribe(self, topics_qos_list):
        if not self.is_connected:
            print(f"Subscriber [{self.client_id}]: Not connected. Cannot subscribe.")
            return
        
        # Ensure topics_qos_list is a list of (topic, qos) tuples
        formatted_topics_qos = []
        for item in topics_qos_list:
            if isinstance(item, str):
                formatted_topics_qos.append((item, 1)) # Default QoS 1 for string topics
            elif isinstance(item, tuple) and len(item) == 2:
                formatted_topics_qos.append(item)
            else:
                print(f"Subscriber [{self.client_id}]: Invalid topic format: {item}. Skipping.")
        
        if not formatted_topics_qos:
            print(f"Subscriber [{self.client_id}]: No valid topics to subscribe.")
            return

        result, mid = self.client.subscribe(formatted_topics_qos)
        if result == mqtt.MQTT_ERR_SUCCESS:
            self._subscribed_topics_map[mid] = [tq[0] for tq in formatted_topics_qos]
            # print(f"Subscriber [{self.client_id}]: Subscribe request sent (MID: {mid})")
        else:
            print(f"Subscriber [{self.client_id}]: Failed to send subscribe request. Error: {mqtt.error_string(result)}")


    def connect(self, topics_to_subscribe):
        """
        Connects to the broker and subscribes to the given topics.
        Args:
            topics_to_subscribe: A list of topic strings or (topic, qos) tuples.
                                 e.g., ["news/updates", ("alerts/critical", 1)]
        """
        if not isinstance(topics_to_subscribe, list):
            topics_to_subscribe = [topics_to_subscribe]
        self._subscribed_topics_on_connect = topics_to_subscribe # Store for _on_connect
        
        try:
            print(f"Subscriber [{self.client_id}]: Attempting to connect...")
            self.client.connect(self.broker_address, self.port, keepalive=60)
            self.client.loop_start() # Non-blocking
        except Exception as e:
            print(f"Subscriber [{self.client_id}]: Connection exception: {e}")
            self.is_connected = False

    def get_messages(self):
        """Retrieves all currently queued messages and clears the queue."""
        messages_batch = []
        while True:
            try:
                messages_batch.append(self._messages.popleft())
            except IndexError: # Queue is empty
                break
        return messages_batch

    def disconnect(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            print(f"Subscriber [{self.client_id}]: Disconnecting.")

# --- Example Usage for BasicMQTTSubscriber ---
if __name__ == "__main__":
    # For the subscriber example to work well with the publisher example,
    # run the publisher first or concurrently.
    
    subscriber = BasicMQTTSubscriber(broker_address="mqtt.eclipseprojects.io")
    # Topics the subscriber will listen to. Can be strings or (topic, qos_level) tuples.
    subscriber.connect(topics_to_subscribe=["game/player/position", ("system/announcements", 0)])

    print("\n--- Reading Frame Data (Example) ---")
    try:
        for _ in range(10): # Simulate reading messages for 10 "frames" or seconds
            if not subscriber.is_connected:
                print("Subscriber trying to reconnect or waiting for connection...")
                time.sleep(1) # Wait if not connected yet.
                continue

            messages = subscriber.get_messages()
            if messages:
                print(f"Frame tick: Received {len(messages)} message(s):")
                for topic, payload in messages:
                    print(f"  Topic: {topic}, Payload: {payload}")
            else:
                print("Frame tick: No new messages.")
            
            time.sleep(1) # Simulate time between frames/checks
    except KeyboardInterrupt:
        print("Subscriber interrupted.")
    finally:
        subscriber.disconnect()
        print("Basic Subscriber example finished.")
