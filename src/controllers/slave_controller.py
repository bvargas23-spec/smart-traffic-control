import json
import time
import logging
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

# AWS IoT configuration
ENDPOINT = "a2ao1owrs8g0lu-ats.iot.us-east-2.amazonaws.com"  # ← Replace this with your actual AWS IoT endpoint
CLIENT_ID = "TrafficLightSlave"
CERT_PATH = "certs/certificate.pem.crt"
PRIVATE_KEY_PATH = "certs/private.pem.key"
ROOT_CA_PATH = "certs/AmazonRootCA1.pem"
TOPIC = "traffic/intersection/timing"  # Topic to receive timing commands

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
        
        self.mqtt_client = self._setup_mqtt_client()
        self.last_command_time = 0
        self.heartbeat_interval = 60
        
        logger.info("Slave controller initialized")
    
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
        mqtt_client.subscribe(TOPIC, 1, self._timing_command_callback)
        return mqtt_client
    
    def _timing_command_callback(self, client, userdata, message):
        """Callback for received timing commands"""
        try:
            payload = json.loads(message.payload)
            logger.info(f"Received timing command: {payload}")
            
            if "timing_plan" in payload:
                timing_plan = {}
                for phase_str, green_time in payload["timing_plan"].items():
                    phase = Phase[phase_str]
                    timing_plan[phase] = green_time
                
                self.signal_controller.update_timing_plan(timing_plan)
                logger.info(f"Updated timing plan: {timing_plan}")
                self.last_command_time = time.time()
        except Exception as e:
            logger.error(f"Error processing timing command: {e}")
    
    def start(self):
        """Start the slave controller"""
        try:
            logger.info(f"Connecting to AWS IoT endpoint: {ENDPOINT}")
            self.mqtt_client.connect()
            logger.info("Connected to AWS IoT")
            self.signal_controller.start_control_loop()
            logger.info("Signal control loop started")
            logger.info("Slave controller running - waiting for commands from master")
            
            while True:
                current_time = time.time()
                if current_time - self.last_command_time > 300:
                    logger.warning("No timing commands received for 5 minutes - using default timing")
                time.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("Slave controller stopped by user")
        except Exception as e:
            logger.error(f"Error in slave controller: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.mqtt_client.disconnect()
            logger.info("Disconnected from AWS IoT")
        except:
            pass
        
        if hasattr(self, 'signal_controller'):
            self.signal_controller.cleanup()
        
        logger.info("Slave controller cleaned up")

# Main entry point
if __name__ == "__main__":
    controller = SlaveController()
    controller.start()
