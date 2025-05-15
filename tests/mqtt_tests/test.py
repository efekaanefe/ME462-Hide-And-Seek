import paho.mqtt.client as mqtt
import time

# --- Configuration ---
broker_address = "mqtt.eclipseprojects.io"  # Replace with your broker address
port = 1883
client_id = "python-mqtt-subscriber-example" # Choose a unique client ID

# --- Callback Functions ---
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"Successfully connected to MQTT broker: {broker_address}")
    else:
        print(f"Failed to connect, reason code: {reason_code}")

def on_disconnect(client, userdata, flags, reason_code, properties):
    print(f"Disconnected from MQTT broker with reason code: {reason_code}")

# --- Client Setup ---
# When creating the client, you can specify the MQTT version.
# paho.mqtt.client.CallbackAPIVersion.VERSION1 is the default.
# For MQTTv5 specific features, you might use paho.mqtt.client.CallbackAPIVersion.VERSION2
# and client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id)
client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv5)
# Or for older callback API (often seen in older examples):
# client = mqtt.Client(client_id)


# Assign callback functions
client.on_connect = on_connect
client.on_disconnect = on_disconnect

# --- Connect to Broker ---
try:
    client.connect(broker_address, port=port, keepalive=60)
except Exception as e:
    print(f"Error connecting to broker: {e}")
    exit()

# Start a non-blocking loop in a separate thread.
# This allows your main program to continue running while the MQTT client handles network traffic.
client.loop_start()

# Keep the main thread alive to allow the client to operate
# In a real application, your main logic would go here.
try:
    while True:
        time.sleep(1)  # Keep the main thread alive
except KeyboardInterrupt:
    print("Exiting...")
finally:
    client.loop_stop() # Stop the network loop
    client.disconnect() # Disconnect from the broker
    print("Cleanly disconnected.")
