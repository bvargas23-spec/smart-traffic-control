# slave_controller.py
import os
import json
import time
import logging
import traceback
from signal_controller import SignalController, Phase, Direction, SignalState
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("slave_controller.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SlaveController")

# AWS IoT configuration - using absolute paths for certificates
ENDPOINT = "a2ao1owrs8g0lu-ats.iot.us-east-2.amazonaws.com"  # ← Replace this with your actual AWS IoT endpoint
CLIENT_ID = "TrafficLightSlave"
CERT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs/certificate.pem.crt")
PRIVATE_KEY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs/private.pem.key")
ROOT_CA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs/AmazonRootCA1.pem")
TIMING_TOPIC = "traffic/intersection/timing"  # Topic to receive timing commands
STATUS_TOPIC = "traffic/slave/status"  # Topic to report slave status
TEST_TOPIC = "traffic/test"  # Topic for test messages

class SlaveController:
    """Slave traffic signal controller that receives commands from the master"""
    
    def __init__(self):
        """Initialize the slave controller"""
        self.signal_controller = SignalController(
            traffic_light_pins={
                Direction.NORTH.value: {"red": "P8_7", "yellow": "P8_8", "green": "P8_9"},
                Direction.SOUTH.value: {"red": "P8_10", "yellow": "P8_11", "green": "P8_12"},
                Direction.EAST.value: {"red": "P8_13", "yellow": "P8_14", "green": "P8_15"},
                Direction.WEST.value: {"red": "P8_16", "yellow": "P8_17", "green": "P8_18"}
            },
            min_green_time=15,
            yellow_time=3,
            all_red_time=2
        )
        
        # Check if certificate files exist
        self._check_certificates()
        
        self.mqtt_client = self._setup_mqtt_client()
        self.last_command_time = 0
        self.heartbeat_interval = 60
        
        logger.info("Slave controller initialized")
    
    def _check_certificates(self):
        """Verify certificate files exist and have content"""
        for path, name in [
            (CERT_PATH, "Certificate"),
            (PRIVATE_KEY_PATH, "Private key"),
            (ROOT_CA_PATH, "Root CA")
        ]:
            if not os.path.exists(path):
                logger.error(f"{name} file not found at: {path}")
            else:
                size = os.path.getsize(path)
                logger.info(f"{name} file exists at {path} ({size} bytes)")
    
    def _setup_mqtt_client(self):
        """Set up and configure the MQTT client"""
        mqtt_client = AWSIoTMQTTClient(CLIENT_ID)
        mqtt_client.configureEndpoint(ENDPOINT, 8883)
        mqtt_client.configureCredentials(ROOT_CA_PATH, PRIVATE_KEY_PATH, CERT_PATH)
        mqtt_client.configureAutoReconnectBackoffTime(1, 32, 20)
        mqtt_client.configureOfflinePublishQueueing(-1)
        mqtt_client.configureDrainingFrequency(2)
        mqtt_client.configureConnectDisconnectTimeout(10)
        mqtt_client.configureMQTTOperationTimeout(5)
        
        # Setup callbacks for connection status
        def on_online_callback():
            logger.info("MQTT Connection established - client is online")
            self._publish_status("connected")
            
        mqtt_client.onOnline = on_online_callback
        
        def on_offline_callback():
            logger.warning("MQTT Connection lost - client is offline")
            
        mqtt_client.onOffline = on_offline_callback
        
        # Setup subscription callback
        mqtt_client.subscribe(TIMING_TOPIC, 1, self._timing_command_callback)
        mqtt_client.subscribe(TEST_TOPIC, 1, self._test_message_callback)
        
        return mqtt_client
    
    def _timing_command_callback(self, client, userdata, message):
        """Callback for received timing commands"""
        try:
            payload = json.loads(message.payload)
            logger.info(f"Received timing command: {payload}")
            
            if "timing_plan" in payload:
                timing_plan = {}
                for phase_str, green_time in payload["timing_plan"].items():
                    try:
                        phase = Phase[phase_str]
                        timing_plan[phase] = green_time
                    except KeyError:
                        logger.error(f"Invalid phase name: {phase_str}")
                
                if timing_plan:
                    self.signal_controller.update_timing_plan(timing_plan)
                    logger.info(f"Updated timing plan: {timing_plan}")
                    self.last_command_time = time.time()
                    self._publish_status("updated", {"timing_plan": timing_plan})
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in message: {message.payload}")
        except Exception as e:
            logger.error(f"Error processing timing command: {e}")
            logger.error(traceback.format_exc())
    
    def _test_message_callback(self, client, userdata, message):
        """Callback for test messages"""
        try:
            payload = json.loads(message.payload)
            logger.info(f"Received test message: {payload}")
            
            # Respond to test message to confirm bidirectional communication
            response = {
                "timestamp": time.time(),
                "client_id": CLIENT_ID,
                "responding_to": payload.get("client_id", "unknown"),
                "message": "Test message received"
            }
            
            self.mqtt_client.publish(TEST_TOPIC + "/response", json.dumps(response), 1)
            logger.info("Responded to test message")
            
        except Exception as e:
            logger.error(f"Error processing test message: {e}")
    
    def _publish_status(self, status, additional_info=None):
        """Publish status information to the status topic"""
        try:
            status_message = {
                "timestamp": time.time(),
                "client_id": CLIENT_ID,
                "status": status
            }
            
            if additional_info:
                status_message.update(additional_info)
                
            self.mqtt_client.publish(STATUS_TOPIC, json.dumps(status_message), 1)
            logger.info(f"Published status: {status}")
            return True
        except Exception as e:
            logger.error(f"Error publishing status: {e}")
            return False
    
    def start(self):
        """Start the slave controller"""
        try:
            logger.info(f"Connecting to AWS IoT endpoint: {ENDPOINT}")
            connect_result = self.mqtt_client.connect()
            
            if not connect_result:
                logger.error("Failed to connect to AWS IoT")
                return
                
            logger.info("Connected to AWS IoT")
            self._publish_status("starting")
            
            self.signal_controller.start_control_loop()
            logger.info("Signal control loop started")
            logger.info("Slave controller running - waiting for commands from master")
            
            self._publish_status("running")
            
            last_heartbeat_time = time.time()
            while True:
                current_time = time.time()
                
                # Check if we've received a command recently
                if current_time - self.last_command_time > 300:
                    logger.warning("No timing commands received for 5 minutes - using default timing")
                
                # Send periodic heartbeat
                if current_time - last_heartbeat_time > self.heartbeat_interval:
                    self._publish_status("heartbeat", {
                        "uptime": current_time - self.last_command_time,
                        "last_command": self.last_command_time
                    })
                    last_heartbeat_time = current_time
                
                time.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("Slave controller stopped by user")
        except Exception as e:
            logger.error(f"Error in slave controller: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self._publish_status("shutting_down")
            self.mqtt_client.disconnect()
            logger.info("Disconnected from AWS IoT")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        if hasattr(self, 'signal_controller'):
            self.signal_controller.cleanup()
        
        logger.info("Slave controller cleaned up")

# Main entry point
if __name__ == "__main__":
    # Print working directory and certificate info for debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"Certificate paths:")
    print(f"  Cert: {CERT_PATH} (exists: {os.path.exists(CERT_PATH)})")
    print(f"  Private key: {PRIVATE_KEY_PATH} (exists: {os.path.exists(PRIVATE_KEY_PATH)})")
    print(f"  Root CA: {ROOT_CA_PATH} (exists: {os.path.exists(ROOT_CA_PATH)})")
    
    controller = SlaveController()
    controller.start()