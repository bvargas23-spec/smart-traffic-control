# master_controller.py - Combines adaptive control with MQTT publishing
import os
import time
import json
import logging
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

# === AWS IoT configuration ===
ENDPOINT = "a2ao1owrs8g0lu-ats.iot.us-east-2.amazonaws.com"  # <-- Replace with your real AWS endpoint
CLIENT_ID = "TrafficLightMaster"
CERT_PATH = "certs/certificate.pem.crt"
PRIVATE_KEY_PATH = "certs/private.pem.key"
ROOT_CA_PATH = "certs/AmazonRootCA1.pem"
TIMING_TOPIC = "traffic/intersection/timing"
STATUS_TOPIC = "traffic/master/status"

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

    def _setup_mqtt_client(self):
        mqtt_client = AWSIoTMQTTClient(CLIENT_ID)
        mqtt_client.configureEndpoint(ENDPOINT, 8883)
        mqtt_client.configureCredentials(ROOT_CA_PATH, PRIVATE_KEY_PATH, CERT_PATH)
        mqtt_client.configureAutoReconnectBackoffTime(1, 32, 20)
        mqtt_client.configureOfflinePublishQueueing(-1)
        mqtt_client.configureDrainingFrequency(2)
        mqtt_client.configureConnectDisconnectTimeout(10)
        mqtt_client.configureMQTTOperationTimeout(5)
        return mqtt_client

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
            self.mqtt_client.publish(TIMING_TOPIC, payload, 1)
            logger.info(f"Published timing plan to {TIMING_TOPIC}")
            return True
        except Exception as e:
            logger.error(f"Error publishing timing plan: {e}")
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
            self.mqtt_client.publish(STATUS_TOPIC, payload, 1)
            logger.info(f"Published status to {STATUS_TOPIC}")
            return True
        except Exception as e:
            logger.error(f"Error publishing status: {e}")
            return False

    def update_timing(self):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        traffic_data = self.fetch_traffic_data()
        if not traffic_data:
            logger.warning("Failed to update timing due to missing traffic data")
            return
        timing_plan = self.calculate_timing_plan(traffic_data)
        self.signal_controller.update_timing_plan(timing_plan)
        self.publish_timing_plan(timing_plan)
        self.publish_status(traffic_data, timing_plan)
        self.last_update_time = current_time
        logger.info("Successfully updated signal timing")

    def start(self):
        try:
            logger.info(f"Connecting to AWS IoT endpoint: {ENDPOINT}")
            self.mqtt_client.connect()
            logger.info("Connected to AWS IoT")
            logger.info("Performing initial traffic data fetch...")
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
        finally:
            self.cleanup()

    def save_data_log(self):
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.data_log, f, indent=2, default=str)
            logger.info(f"Saved data log to {self.log_file}")
        except Exception as e:
            logger.error(f"Error saving data log: {e}")

    def cleanup(self):
        self.save_data_log()
        try:
            self.mqtt_client.disconnect()
            logger.info("Disconnected from AWS IoT")
        except:
            pass
        if hasattr(self, 'signal_controller'):
            self.signal_controller.cleanup()
        logger.info("Master controller cleaned up")

# === Entry Point ===
if __name__ == "__main__":
    intersection_lat = 33.960192828395996
    intersection_lon = -84.52790520126695
    controller = MasterController(
        intersection_lat=intersection_lat,
        intersection_lon=intersection_lon,
        update_interval=300
    )
    controller.start()
