# test_mqtt_publish.py - Simple test script for AWS IoT MQTT publishing with async publish + debug logging
import os
import json
import time
import traceback
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

# Configure logging (full SDK debug logs)
import logging
logging.getLogger("AWSIoTPythonSDK").setLevel(logging.DEBUG)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mqtt_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MQTTTest")

# AWS IoT configuration - using absolute paths for certificates
ENDPOINT = "a2ao1owrs8g0lu-ats.iot.us-east-2.amazonaws.com"
CLIENT_ID = "TestPublisher"
CERT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs/certificate.pem.crt")
PRIVATE_KEY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs/private.pem.key")
ROOT_CA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs/AmazonRootCA1.pem")
TEST_TOPIC = "traffic/test"

def check_certificates():
    """Verify certificate files exist and have content"""
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {os.path.abspath(__file__)}")
    
    for path, name in [
        (CERT_PATH, "Certificate"),
        (PRIVATE_KEY_PATH, "Private key"),
        (ROOT_CA_PATH, "Root CA")
    ]:
        if not os.path.exists(path):
            print(f"❌ {name} file not found at: {path}")
            return False
        else:
            size = os.path.getsize(path)
            print(f"✅ {name} file exists at {path} ({size} bytes)")
    
    return True

def message_callback(client, userdata, message):
    """Callback for received messages"""
    try:
        print(f"Received message on {message.topic}: {message.payload}")
        payload_str = message.payload.decode('utf-8')
        payload = json.loads(payload_str)
        print(f"Decoded JSON: {json.dumps(payload, indent=2)}")
    except Exception as e:
        print(f"Error processing message: {e}")

def setup_mqtt_client():
    """Set up and configure the MQTT client"""
    mqtt_client = AWSIoTMQTTClient(CLIENT_ID)
    mqtt_client.configureEndpoint(ENDPOINT, 8883)
    mqtt_client.configureCredentials(ROOT_CA_PATH, PRIVATE_KEY_PATH, CERT_PATH)
    mqtt_client.configureAutoReconnectBackoffTime(1, 32, 20)
    mqtt_client.configureOfflinePublishQueueing(-1)
    mqtt_client.configureDrainingFrequency(2)
    mqtt_client.configureConnectDisconnectTimeout(10)
    mqtt_client.configureMQTTOperationTimeout(5)

    # Set up callbacks
    def on_online_callback():
        print("MQTT Connection established - client is online")
    
    mqtt_client.onOnline = on_online_callback
    
    def on_offline_callback():
        print("MQTT Connection lost - client is offline")
    
    mqtt_client.onOffline = on_offline_callback
    
    # Subscribe to topics for bi-directional testing
    mqtt_client.subscribe(TEST_TOPIC, 1, message_callback)
    mqtt_client.subscribe(f"{TEST_TOPIC}/response", 1, message_callback)
    mqtt_client.subscribe("traffic/intersection/timing", 1, message_callback)
    
    return mqtt_client

def test_publish():
    """Test publishing to AWS IoT using async publish"""
    if not check_certificates():
        print("Certificate check failed - please fix certificate issues before continuing")
        return False

    mqtt_client = setup_mqtt_client()
    print("Connecting to AWS IoT...")

    try:
        connect_result = mqtt_client.connect()
        print(f"Connect result: {connect_result}")

        if connect_result:
            print("Connected successfully. Waiting for MQTT subscriptions to finalize...")
            time.sleep(6)

            # Async publish ACK callback
            def puback_callback(mid):
                print(f"✅ Publish ACK received for message ID: {mid}")

            print("Now publishing test message...")

            message = {
                "timestamp": time.time(),
                "client_id": CLIENT_ID,
                "test_message": "Hello from test publisher",
                "test_number": 1
            }
            payload = json.dumps(message)

            # Async publish
            mqtt_client.publishAsync(TEST_TOPIC, payload, 1, ackCallback=puback_callback)

            timing_message = {
                "timestamp": time.time(),
                "intersection_id": "test_publisher",
                "timing_plan": {
                    "NORTH_SOUTH": 30,
                    "EAST_WEST": 30
                }
            }

            mqtt_client.publishAsync("traffic/intersection/timing", json.dumps(timing_message), 1, ackCallback=puback_callback)

            print("Waiting for 10 seconds to receive ACKs and possible responses...")
            time.sleep(10)

            mqtt_client.disconnect()
            print("Disconnected")
            return True
        else:
            print("Failed to connect")
            return False
    except Exception as e:
        print(f"Error during test: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_publish()
