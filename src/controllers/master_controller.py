# master_controller.py - Combines adaptive control with MQTT publishing
import os
import time
import json
import logging
import traceback
from datetime import datetime
from signal_controller import SignalController, Phase, Direction, SignalState
from src.api.tomtom_client import TomTomClient
from src.api.data_processor import DataProcessor
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("master_controller.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MasterController")

# AWS IoT configuration
ENDPOINT = "a2ao1owrs8g0lu-ats.iot.us-east-2.amazonaws.com"
CLIENT_ID = "TrafficLightMaster"
CERT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs/certificate.pem.crt")
PRIVATE_KEY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs/private.pem.key")
ROOT_CA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs/AmazonRootCA1.pem")
TIMING_TOPIC = "traffic/intersection/timing"
STATUS_TOPIC = "traffic/master/status"
TEST_TOPIC = "traffic/test"

class MasterController:
    def __init__(self, intersection_lat, intersection_lon, update_interval=300):
        api_key = os.getenv("TOMTOM_API_KEY")
        if not api_key:
            raise ValueError("No TomTom API key found in environment variables")
        self.tomtom_client = TomTomClient(api_key=api_key)
        self.data_processor = DataProcessor()
        self.intersection_lat = intersection_lat
        self.intersection_lon = intersection_lon
        self.direction_mapping = {
            Direction.NORTH: "North",
            Direction.SOUTH: "South",
            Direction.EAST: "East",
            Direction.WEST: "West"
        }
        self.update_interval = update_interval
        self.last_update_time = 0
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
        self.data_log = []
        self.log_file = "traffic_data_log.json"
        logger.info(f"Master controller initialized for intersection at {intersection_lat}, {intersection_lon}")
        self._check_certificates()

    def _check_certificates(self):
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
        mqtt_client = AWSIoTMQTTClient(CLIENT_ID)
        mqtt_client.configureEndpoint(ENDPOINT, 8883)
        mqtt_client.configureCredentials(ROOT_CA_PATH, PRIVATE_KEY_PATH, CERT_PATH)
        mqtt_client.configureAutoReconnectBackoffTime(1, 32, 20)
        mqtt_client.configureOfflinePublishQueueing(-1)
        mqtt_client.configureDrainingFrequency(2)
        mqtt_client.configureConnectDisconnectTimeout(10)
        mqtt_client.configureMQTTOperationTimeout(5)

        def on_online_callback():
            logger.info("MQTT Connection established - client is online")
            self._send_test_message()

        def on_offline_callback():
            logger.warning("MQTT Connection lost - client is offline")

        mqtt_client.onOnline = on_online_callback
        mqtt_client.onOffline = on_offline_callback
        return mqtt_client

    def _send_test_message(self):
        try:
            test_message = {
                "timestamp": datetime.now().isoformat(),
                "message": "Master controller test message",
                "client_id": CLIENT_ID
            }
            payload = json.dumps(test_message)
            logger.info(f"Sending test message to {TEST_TOPIC}: {payload}")
            result = self.mqtt_client.publish(TEST_TOPIC, payload, 1)
            logger.info(f"Test message publish result: {result}")
        except Exception as e:
            logger.error(f"Error sending test message: {e}")
            logger.error(traceback.format_exc())

    def fetch_traffic_data(self):
        try:
            traffic_summary = self.tomtom_client.get_traffic_summary(self.intersection_lat, self.intersection_lon)
            processed_data = self.data_processor.process_intersection_data({
                "intersection": traffic_summary["intersection"],
                "approaches": traffic_summary["approaches"]
            })
            logger.info(f"Fetched traffic data with {len(processed_data['approaches'])} approaches")
            self.data_log.append({
                "timestamp": datetime.now().isoformat(),
                "data": processed_data
            })
            if len(self.data_log) % 10 == 0:
                self.save_data_log()
            return processed_data
        except Exception as e:
            logger.error(f"Error fetching traffic data: {e}")
            logger.error(traceback.format_exc())
            return None

    def calculate_timing_plan(self, traffic_data):
        if not traffic_data or "error" in traffic_data:
            logger.warning("Using default timing due to missing traffic data")
            return {
                Phase.NORTH_SOUTH: 30,
                Phase.EAST_WEST: 30
            }
        optimal_timing = self.data_processor.calculate_optimal_cycle_times(traffic_data)
        phase_times = optimal_timing.get("phase_times", {})
        timing_plan = {
            Phase.NORTH_SOUTH: phase_times.get("North-South", {}).get("green_time", 30),
            Phase.EAST_WEST: phase_times.get("East-West", {}).get("green_time", 30)
        }
        logger.info(f"Calculated timing plan: NS={timing_plan[Phase.NORTH_SOUTH]}s, EW={timing_plan[Phase.EAST_WEST]}s")
        return timing_plan

    def publish_timing_plan(self, timing_plan):
        try:
            serializable_plan = {phase.name: green_time for phase, green_time in timing_plan.items()}
            message = {
                "timestamp": datetime.now().isoformat(),
                "intersection_id": "master",
                "timing_plan": serializable_plan
            }
            payload = json.dumps(message)
            logger.info(f"Attempting to publish to {TIMING_TOPIC}: {payload}")
            result = self.mqtt_client.publish(TIMING_TOPIC, payload, 1)
            logger.info(f"Publish result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error publishing timing plan: {e}")
            logger.error(traceback.format_exc())
            return False

    def publish_status(self, traffic_data=None, timing_plan=None):
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "intersection_id": "master",
                "status": "operational"
            }
            if traffic_data:
                status["traffic"] = {
                    "weighted_score": traffic_data.get("intersection", {}).get("weighted_score"),
                    "most_congested": traffic_data.get("intersection", {}).get("most_congested_approach")
                }
            if timing_plan:
                status["timing"] = {phase.name: green_time for phase, green_time in timing_plan.items()}
            payload = json.dumps(status)
            logger.info(f"Attempting to publish status to {STATUS_TOPIC}")
            result = self.mqtt_client.publish(STATUS_TOPIC, payload, 1)
            logger.info(f"Status publish result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error publishing status: {e}")
            logger.error(traceback.format_exc())
            return False

    def update_timing(self):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        logger.info("Time to update timing based on traffic data")
        traffic_data = self.fetch_traffic_data()
        if not traffic_data:
            logger.warning("Failed to update timing due to missing traffic data")
            return

        timing_plan = self.calculate_timing_plan(traffic_data)
        self.signal_controller.update_timing_plan(timing_plan)
        self.publish_timing_plan(timing_plan)
        self.publish_status(traffic_data, timing_plan)
        self.last_update_time = current_time

    def start(self):
        try:
            logger.info(f"Connecting to AWS IoT endpoint: {ENDPOINT}")
            connect_result = self.mqtt_client.connect()
            if not connect_result:
                logger.error("Failed to connect to AWS IoT")
                return

            logger.info("Connected to AWS IoT")
            self._send_test_message()
            traffic_data = self.fetch_traffic_data()
            if traffic_data:
                timing_plan = self.calculate_timing_plan(traffic_data)
                self.signal_controller.update_timing_plan(timing_plan)
                self.publish_timing_plan(timing_plan)
                self.publish_status(traffic_data, timing_plan)

            self.signal_controller.start_control_loop()
            logger.info(f"Starting main control loop with {self.update_interval}s update interval")
            self.last_update_time = time.time()

            while True:
                self.update_timing()
                time.sleep(10)
        except KeyboardInterrupt:
            logger.info("Master controller stopped by user")
        except Exception as e:
            logger.error(f"Error in master controller: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.cleanup()

    def save_data_log(self):
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.data_log, f, indent=2, default=str)
            logger.info(f"Saved data log to {self.log_file}")
        except Exception as e:
            logger.error(f"Error saving data log: {e}")
            logger.error(traceback.format_exc())

    def cleanup(self):
        self.save_data_log()
        try:
            self.mqtt_client.disconnect()
            logger.info("Disconnected from AWS IoT")
        except Exception as e:
            logger.error(f"Error disconnecting from AWS IoT: {e}")

        if hasattr(self, 'signal_controller'):
            self.signal_controller.cleanup()
        logger.info("Master controller cleaned up")

if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    print("Certificate paths:")
    print(f"  Cert: {CERT_PATH} (exists: {os.path.exists(CERT_PATH)})")
    print(f"  Private key: {PRIVATE_KEY_PATH} (exists: {os.path.exists(PRIVATE_KEY_PATH)})")
    print(f"  Root CA: {ROOT_CA_PATH} (exists: {os.path.exists(ROOT_CA_PATH)})")

    intersection_lat = 33.960192828395996
    intersection_lon = -84.52790520126695
    controller = MasterController(
        intersection_lat=intersection_lat,
        intersection_lon=intersection_lon,
        update_interval=300
    )
    controller.start()