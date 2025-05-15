import paho.mqtt.client as mqtt
import time

broker_address = "mqtt.eclipseprojects.io"
port = 1883
topic_to_subscribe = "my/test/topic" # Ensure this is the topic you are subscribing to
client_id_subscriber = "python-mqtt-subscriber-example-receiver-fixed"

# --- Callback Functions for Subscriber ---
def on_connect_subscriber(client, userdata, flags, reason_code, properties):
    # The 'reason_code' here is a paho.mqtt.reasoncodes.ReasonCode object for the CONNECT operation
    if reason_code == mqtt.CONNACK_ACCEPTED: # You can also check reason_code.is_failure
        print(f"Subscriber connected to MQTT broker: {broker_address}")
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        # The subscribe call below uses the global 'topic_to_subscribe'
        client.subscribe(topic_to_subscribe, qos=1)
    else:
        print(f"Subscriber failed to connect, reason code: {reason_code}")

def on_message(client, userdata, msg):
    """The callback for when a PUBLISH message is received from the server."""
    print(f"Received message on topic '{msg.topic}': {msg.payload.decode()}")

def on_subscribe(client, userdata, mid, reason_codes, properties):
    """
    Called when the broker responds to a subscribe request.
    - mid: matches the mid variable returned by the subscribe() call.
    - reason_codes: a list of ReasonCode objects. For a single topic subscription,
                    this list will contain one ReasonCode.
    - properties: The MQTTv5 properties, if any.
    """
    if reason_codes and len(reason_codes) > 0:
        rc = reason_codes[0]  # Get the first (and in this case, only) ReasonCode
        if not rc.is_failure:
            # rc.value will contain the granted QoS level for a successful subscription (e.g., 0, 1, 2)
            print(f"Successfully subscribed to topic '{topic_to_subscribe}' (MID: {mid}) with QoS {rc.value}")
        else:
            # str(rc) will give a human-readable error name (e.g., "Unspecified error")
            # rc.value will give the numerical error code (e.g., 0x80 for Unspecified error)
            print(f"Failed to subscribe to topic '{topic_to_subscribe}' (MID: {mid}): {str(rc)} (Code: {rc.value})")
    else:
        # This case should ideally not happen if a SUBACK is received for a SUBSCRIBE.
        print(f"on_subscribe (MID: {mid}): Received empty or invalid reason_codes list.")

# --- Subscriber Client Setup ---
# Explicitly use CallbackAPIVersion.VERSION2
subscriber_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id_subscriber, protocol=mqtt.MQTTv5)
subscriber_client.on_connect = on_connect_subscriber
subscriber_client.on_message = on_message
subscriber_client.on_subscribe = on_subscribe

# --- Connect and Loop ---
try:
    print(f"Attempting to connect to broker: {broker_address}")
    subscriber_client.connect(broker_address, port=port, keepalive=60)
except Exception as e:
    print(f"Error connecting subscriber: {e}")
    exit()

print(f"Starting MQTT loop and waiting for messages on topic: {topic_to_subscribe}")
try:
    subscriber_client.loop_forever() # This will block until client.disconnect() is called or an error occurs.
except KeyboardInterrupt:
    print("Subscriber exiting...")
finally:
    subscriber_client.disconnect()
    print("Subscriber cleanly disconnected.")
