import paho.mqtt.client as mqtt
import time

broker_address = "mqtt.eclipseprojects.io"
port = 1883
topic_to_publish = "my/test/topic"
client_id_publisher = "python-mqtt-publisher-example-fixed"

def on_connect_publisher(client, userdata, flags, reason_code, properties):
    # reason_code is a paho.mqtt.reasoncodes.ReasonCode object for connect
    if reason_code == mqtt.CONNACK_ACCEPTED: # or not reason_code.is_failure
        print(f"Publisher connected to MQTT broker: {broker_address}")
    else:
        print(f"Publisher failed to connect, reason code: {reason_code}")

def on_publish(client, userdata, mid, reason_code, properties):
    """
    Callback when a message that was published has been acknowledged by the broker.
    For QoS 1, this means PUBACK was received.
    For QoS 2, this means PUBCOMP was received.
    For QoS 0, this callback is not typically invoked by PUBACK but might be called locally by Paho
    immediately after sending (behavior can vary, check Paho docs for specifics on QoS 0 on_publish).
    In Paho MQTT Python client with CallbackAPIVersion.VERSION2:
    - mid: The message ID for the published message.
    - reason_code: For MQTTv5, this is a ReasonCode object from the PUBACK/PUBCOMP.
                   For MQTTv3.1.1 brokers or if no reason code is in PUBACK, this might be an int (typically 0 for success).
                   If publishing QoS 0, this argument might not be a ReasonCode object from the broker.
    - properties: The MQTTv5 properties from the PUBACK/PUBCOMP, if any.
    """
    # Check if reason_code is an int (likely from MQTTv3.1.1 ack or simple success)
    if isinstance(reason_code, int):
        if reason_code == mqtt.MQTT_ERR_SUCCESS: # MQTT_ERR_SUCCESS is 0
            print(f"Message {mid} published successfully (simple ACK).")
        else:
            # This case might occur if an older version of Paho or a specific configuration
            # passes an integer error code here.
            print(f"Message {mid} publish acknowledged with integer code: {reason_code}")
    elif hasattr(reason_code, 'is_failure'): # It's a ReasonCode object (MQTTv5)
        if not reason_code.is_failure:
            # For successful PUBACK/PUBCOMP in MQTTv5, reason_code.value is typically 0 (Success)
            # or 16 (No matching subscribers - this is a success for the publish itself).
            print(f"Message {mid} published successfully. Reason: {str(reason_code)} (Code: {reason_code.value})")
        else:
            print(f"Failed to publish message {mid}. Reason: {str(reason_code)} (Code: {reason_code.value})")
    else:
        # Fallback for any other unexpected type of reason_code
        print(f"Message {mid} published. Acknowledgement status: {reason_code}")


# --- Publisher Client Setup ---
# Explicitly use CallbackAPIVersion.VERSION2
publisher_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id_publisher, protocol=mqtt.MQTTv5)
publisher_client.on_connect = on_connect_publisher
publisher_client.on_publish = on_publish

try:
    publisher_client.connect(broker_address, port=port, keepalive=60)
except Exception as e:
    print(f"Error connecting publisher: {e}")
    exit()

publisher_client.loop_start()

# --- Publish Messages ---
message_count = 0
try:
    while message_count < 5: # Publish a few messages
        time.sleep(2)
        message = f"Hello MQTT! Message number {message_count}"
        # For QoS 1 or 2, on_publish will be called upon receiving PUBACK/PUBCOMP.
        # For QoS 0, on_publish might be called immediately by the client (behavior can vary).
        result = publisher_client.publish(topic_to_publish, payload=message, qos=1)
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"Sent: '{message}' to topic '{topic_to_publish}' (MID: {result.mid}) - waiting for PUBACK")
        else:
            print(f"Failed to send message to topic {topic_to_publish}, error code: {result.rc}")
        message_count += 1
except KeyboardInterrupt:
    print("Publisher exiting...")
finally:
    publisher_client.loop_stop()
    publisher_client.disconnect()
    print("Publisher cleanly disconnected.")
