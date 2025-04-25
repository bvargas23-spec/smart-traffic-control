#!/usr/bin/env python3
# intersection_coordinator.py - Intelligent traffic light coordinator for connected intersections
import os
import time
import json
import logging
import traceback
from datetime import datetime
from enum import Enum
import threading
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
        logging.FileHandler("intersection_coordinator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("IntersectionCoordinator")

# Enums for phases and directions (copied from signal_controller.py)
class Phase(Enum):
    NORTH_SOUTH = "NORTH_SOUTH"
    EAST_WEST = "EAST_WEST"

class Direction(Enum):
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"

class SignalState(Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"

# AWS IoT configuration
ENDPOINT = "a2ao1owrs8g0lu-ats.iot.us-east-2.amazonaws.com"
CERT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs/certificate.pem.crt")
PRIVATE_KEY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs/private.pem.key")
ROOT_CA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs/AmazonRootCA1.pem")

# MQTT Topics
TIMING_TOPIC = "traffic/intersection/timing"
STATUS_TOPIC = "traffic/intersection/status"
CONGESTION_TOPIC = "traffic/alerts"  # Will be used as prefix
TEST_TOPIC = "traffic/test"

# Intersection definitions from Enhanced_Traffic.py
INTERSECTIONS = {
    "Roswell & Hwy 41": {
        "id": "roswell_hwy41",
        "coordinates": {
            "latitude": 33.960192828395996,
            "longitude": -84.52790520126695
        },
        "approaches": {
            "North": (33.95177778, -84.52108333),
            "South": (33.95030556, -84.52027778),
            "East": (33.95091667, -84.51977778),
            "West": (33.95097222, -84.52163889)
        },
        "speed_limit": 45  # mph
    },
    "Cobb Pkwy & Hwy 120": {
        "id": "cobb_hwy120",
        "coordinates": {
            "latitude": 33.94178,
            "longitude": -84.51568
        },
        "approaches": {
            "North": (33.94230, -84.51568),
            "South": (33.94125, -84.51568),
            "East": (33.94178, -84.51470),
            "West": (33.94178, -84.51670)
        },
        "speed_limit": 45  # mph
    }
}

# Directional relationships between intersections (which directions connect them)
DIRECTIONAL_RELATIONSHIPS = {
    "roswell_hwy41": {
        "South": "cobb_hwy120"  # Going south from Roswell connects to Cobb
    },
    "cobb_hwy120": {
        "North": "roswell_hwy41"  # Going north from Cobb connects to Roswell
    }
}

# Mock signal controller for testing without hardware
class MockSignalController:
    def __init__(self):
        self.current_phase = Phase.NORTH_SOUTH
        self.timing_plan = {
            Phase.NORTH_SOUTH: 30,
            Phase.EAST_WEST: 30
        }
        self.running = False
        self.thread = None
        
    def update_timing_plan(self, timing_plan):
        """Update the signal timing plan"""
        logger.info(f"Updating timing plan: {timing_plan}")
        self.timing_plan = timing_plan
        return True
        
    def get_timing_plan(self):
        """Get the current timing plan"""
        return self.timing_plan
        
    def start_control_loop(self):
        """Start the signal control loop in a background thread"""
        self.running = True
        self.thread = threading.Thread(target=self._control_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Signal control loop started in background thread")
        
    def _control_loop(self):
        """Main control loop for signal state management"""
        while self.running:
            # Switch between phases
            if self.current_phase == Phase.NORTH_SOUTH:
                time.sleep(self.timing_plan.get(Phase.NORTH_SOUTH, 30))
                self.current_phase = Phase.EAST_WEST
            else:
                time.sleep(self.timing_plan.get(Phase.EAST_WEST, 30))
                self.current_phase = Phase.NORTH_SOUTH
                
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        logger.info("Signal controller cleaned up")

class IntersectionCoordinator:
    """
    Coordinates traffic signal timing between multiple intersections
    """
    
    def __init__(self, intersection_name, role="master", update_interval=60, use_mock=True):
        """
        Initialize the intersection coordinator
        
        Args:
            intersection_name (str): Name of the intersection
            role (str): Role of this coordinator ("master" or "responder")
            update_interval (int): Seconds between traffic data updates
            use_mock (bool): Use mock signal controller instead of hardware
        """
        # Verify the intersection exists in our definitions
        if intersection_name not in INTERSECTIONS:
            raise ValueError(f"Unknown intersection: {intersection_name}")
            
        self.intersection_name = intersection_name
        self.intersection_info = INTERSECTIONS[intersection_name]
        self.intersection_id = self.intersection_info["id"]
        self.role = role
        self.update_interval = update_interval
        self.use_mock = use_mock
        
        # Initialize TomTom API client (needed for both roles)
        api_key = os.getenv("TOMTOM_API_KEY")
        if not api_key:
            raise ValueError("No TomTom API key found in environment variables")
        
        self.tomtom_client = TomTomClient(api_key=api_key)
        self.data_processor = DataProcessor()
        
        # Set up the MQTT client with the intersection ID as the client ID
        self.client_id = f"{self.intersection_id}_{role}"
        self.mqtt_client = self._setup_mqtt_client()
        
        # Initialize the signal controller (mock or real)
        if use_mock:
            logger.info("Using mock signal controller")
            self.signal_controller = MockSignalController()
        else:
            # Import the real signal controller
            try:
                from src.controllers.signal_controller import SignalController
                logger.info("Using real signal controller")
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
            except ImportError:
                logger.warning("Could not import SignalController, falling back to mock")
                self.signal_controller = MockSignalController()
                self.use_mock = True
        
        # Keep track of traffic conditions and congestion alerts
        self.traffic_conditions = {}
        self.congestion_alerts = {}
        self.timing_adjustments = {}
        
        # Timing tracking
        self.last_update_time = 0
        self.last_status_time = 0
        
        # Data logging
        self.data_log = []
        self.log_file = f"{self.intersection_id}_data_log.json"
        
        # Check certificates
        self._check_certificates()
        
        logger.info(f"Intersection coordinator initialized for {intersection_name} in {role} role")
    
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
        mqtt_client = AWSIoTMQTTClient(self.client_id)
        mqtt_client.configureEndpoint(ENDPOINT, 8883)
        mqtt_client.configureCredentials(ROOT_CA_PATH, PRIVATE_KEY_PATH, CERT_PATH)
        mqtt_client.configureAutoReconnectBackoffTime(1, 32, 20)
        mqtt_client.configureOfflinePublishQueueing(-1)
        mqtt_client.configureDrainingFrequency(2)
        mqtt_client.configureConnectDisconnectTimeout(10)
        mqtt_client.configureMQTTOperationTimeout(5)
        
        # Set up callbacks for connection status
        def on_online_callback():
            logger.info("MQTT Connection established - client is online")
            self._publish_status("connected")
            
        def on_offline_callback():
            logger.warning("MQTT Connection lost - client is offline")
            
        mqtt_client.onOnline = on_online_callback
        mqtt_client.onOffline = on_offline_callback
        
        # Subscribe to topics based on role
        # Both roles subscribe to congestion alerts
        relationships = DIRECTIONAL_RELATIONSHIPS.get(self.intersection_id, {})
        for direction, target_id in relationships.items():
            topic = f"{CONGESTION_TOPIC}/{target_id}/{self._opposite_direction(direction)}"
            logger.info(f"Subscribing to congestion topic: {topic}")
            mqtt_client.subscribe(topic, 1, self._congestion_alert_callback)
        
        # Subscribe to test topic for debugging
        mqtt_client.subscribe(TEST_TOPIC, 1, self._test_message_callback)
        
        # Role-specific subscriptions
        if self.role == "master":
            # Master sends timing commands and receives status updates
            mqtt_client.subscribe(f"{STATUS_TOPIC}/#", 1, self._status_callback)
        else:  # responder
            # Responder receives timing commands
            mqtt_client.subscribe(f"{TIMING_TOPIC}/{self.intersection_id}", 1, self._timing_command_callback)
        
        return mqtt_client
    
    def _opposite_direction(self, direction):
        """Get the opposite cardinal direction"""
        opposites = {
            "North": "South",
            "South": "North",
            "East": "West",
            "West": "East"
        }
        return opposites.get(direction, direction)
    
    def _timing_command_callback(self, client, userdata, message):
        """Callback for received timing commands (responder only)"""
        try:
            payload = json.loads(message.payload)
            logger.info(f"Received timing command: {payload}")
            
            if "timing_plan" in payload:
                timing_plan = {}
                
                # Convert string phase names to Phase enum
                for phase_str, green_time in payload["timing_plan"].items():
                    try:
                        phase = Phase[phase_str]
                        timing_plan[phase] = green_time
                    except (KeyError, ValueError) as e:
                        logger.error(f"Invalid phase name: {phase_str} - {e}")
                
                if timing_plan:
                    # Update the signal controller
                    self.signal_controller.update_timing_plan(timing_plan)
                    logger.info(f"Updated timing plan: {timing_plan}")
                    
                    # Publish status
                    self._publish_status("updated", {
                        "timing_plan": {k.name: v for k, v in timing_plan.items()}
                    })
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in message: {message.payload}")
        except Exception as e:
            logger.error(f"Error processing timing command: {e}")
            logger.error(traceback.format_exc())
    
    def _status_callback(self, client, userdata, message):
        """Callback for status messages from responders (master only)"""
        try:
            payload = json.loads(message.payload)
            source_id = payload.get("intersection_id", "unknown")
            status = payload.get("status", "unknown")
            logger.info(f"Received status from {source_id}: {status}")
            
            # Track status updates from responders
            if source_id != self.intersection_id:
                self._record_status_update(source_id, status, payload)
                
        except Exception as e:
            logger.error(f"Error processing status message: {e}")
    
    def _record_status_update(self, source_id, status, data):
        """Record status update from a responder"""
        # This could be expanded to track responder health, etc.
        pass
    
    def _congestion_alert_callback(self, client, userdata, message):
        """Callback for congestion alerts from other intersections"""
        try:
            payload = json.loads(message.payload)
            logger.info(f"Received congestion alert: {payload}")
            
            # Extract alert details
            source_id = payload.get("intersection_id")
            direction = payload.get("direction")
            congestion_level = payload.get("congestion_level")
            timestamp = payload.get("timestamp")
            
            if not all([source_id, direction, congestion_level, timestamp]):
                logger.warning(f"Incomplete congestion alert: {payload}")
                return
            
            # Store the alert
            alert_key = f"{source_id}_{direction}"
            self.congestion_alerts[alert_key] = {
                "source_id": source_id,
                "direction": direction,
                "congestion_level": congestion_level,
                "timestamp": timestamp,
                "received_at": time.time()
            }
            
            # Process the alert based on role
            if self.role == "master":
                logger.info(f"Master received congestion alert from {source_id} for {direction} direction")
                # Masters can adjust their own timing or report to dashboard
                self._adjust_timing_for_congestion_alert(source_id, direction, congestion_level)
            else:
                logger.info(f"Responder received congestion alert from {source_id} for {direction} direction")
                # Responders adjust their timing based on upstream congestion
                self._adjust_timing_for_congestion_alert(source_id, direction, congestion_level)
                
        except Exception as e:
            logger.error(f"Error processing congestion alert: {e}")
            logger.error(traceback.format_exc())
    
    def _adjust_timing_for_congestion_alert(self, source_id, direction, congestion_level):
        """
        Adjust timing based on congestion alert from another intersection
        
        Args:
            source_id (str): ID of the intersection that sent the alert
            direction (str): Direction of congestion (North, South, East, West)
            congestion_level (str): Congestion level (Free Flow, Light, Moderate, Heavy, Severe)
        """
        # Determine if we should respond to this alert
        relationships = DIRECTIONAL_RELATIONSHIPS.get(self.intersection_id, {})
        upstream_direction = None
        
        for my_dir, target_id in relationships.items():
            if target_id == source_id and self._opposite_direction(my_dir) == direction:
                upstream_direction = my_dir
                break
        
        if not upstream_direction:
            logger.info(f"No relationship with {source_id} in {direction} direction, ignoring alert")
            return
            
        logger.info(f"Identified relationship: {source_id} affects our {upstream_direction} approach")
        
        # Map the direction to the corresponding phase
        phase_mapping = {
            "North": Phase.NORTH_SOUTH,
            "South": Phase.NORTH_SOUTH,
            "East": Phase.EAST_WEST,
            "West": Phase.EAST_WEST
        }
        
        phase = phase_mapping.get(upstream_direction)
        if not phase:
            logger.warning(f"Unknown direction: {upstream_direction}")
            return
        
        # Calculate adjustment based on congestion level
        adjustment = 0
        if congestion_level == "Moderate":
            adjustment = 5  # Add 5 seconds for moderate congestion
        elif congestion_level == "Heavy":
            adjustment = 10  # Add 10 seconds for heavy congestion
        elif congestion_level == "Severe":
            adjustment = 15  # Add 15 seconds for severe congestion
        
        if adjustment > 0:
            logger.info(f"Adjusting {phase.name} timing by +{adjustment}s due to {congestion_level} congestion")
            
            # Get current timing plan
            current_plan = self.signal_controller.get_timing_plan()
            
            # Apply adjustment (extend green time for the affected phase)
            new_plan = current_plan.copy()
            new_plan[phase] = min(60, current_plan.get(phase, 30) + adjustment)  # Cap at 60 seconds
            
            # Update the signal controller
            self.signal_controller.update_timing_plan(new_plan)
            
            # Track the adjustment
            adjustment_key = f"{source_id}_{direction}"
            self.timing_adjustments[adjustment_key] = {
                "source_id": source_id,
                "direction": direction,
                "congestion_level": congestion_level,
                "adjustment": adjustment,
                "adjusted_at": time.time(),
                "new_timing": new_plan[phase]
            }
            
            # Publish status update
            self._publish_status("adjusted", {
                "source_id": source_id,
                "direction": direction,
                "adjustment": adjustment,
                "timing_plan": {k.name: v for k, v in new_plan.items()}
            })
    
    def _test_message_callback(self, client, userdata, message):
        """Callback for test messages"""
        try:
            payload = json.loads(message.payload)
            logger.info(f"Received test message: {payload}")
            
            # Respond to test message to confirm bidirectional communication
            response = {
                "timestamp": time.time(),
                "intersection_id": self.intersection_id,
                "responding_to": payload.get("client_id", "unknown"),
                "message": "Test message received"
            }
            
            self.mqtt_client.publish(f"{TEST_TOPIC}/response", json.dumps(response), 1)
            
        except Exception as e:
            logger.error(f"Error processing test message: {e}")
    
    def _publish_status(self, status, additional_info=None):
        """Publish status information to the status topic"""
        try:
            status_message = {
                "timestamp": time.time(),
                "intersection_id": self.intersection_id,
                "intersection_name": self.intersection_name,
                "role": self.role,
                "status": status
            }
            
            # Add any additional info
            if additional_info:
                status_message.update(additional_info)
                
            # Publish the status message
            topic = f"{STATUS_TOPIC}/{self.intersection_id}"
            payload = json.dumps(status_message)
            logger.info(f"Publishing status to {topic}: {status}")
            self.mqtt_client.publish(topic, payload, 1)
            return True
        except Exception as e:
            logger.error(f"Error publishing status: {e}")
            return False
    
    def _publish_timing_plan(self, timing_plan):
        """Publish timing plan to responders (master only)"""
        try:
            # Convert Phase objects to strings for JSON serialization
            serializable_plan = {phase.name: green_time for phase, green_time in timing_plan.items()}
            message = {
                "timestamp": time.time(),
                "intersection_id": self.intersection_id,
                "intersection_name": self.intersection_name,
                "timing_plan": serializable_plan
            }
            
            # Publish to the timing topic
            topic = f"{TIMING_TOPIC}/{self.intersection_id}"
            payload = json.dumps(message)
            logger.info(f"Publishing timing plan to {topic}")
            self.mqtt_client.publish(topic, payload, 1)
            return True
        except Exception as e:
            logger.error(f"Error publishing timing plan: {e}")
            return False
    
    def _publish_congestion_alert(self, direction, congestion_level):
        """
        Publish congestion alert for a specific direction
        
        Args:
            direction (str): Direction of congestion (North, South, East, West)
            congestion_level (str): Congestion level (Free Flow, Light, Moderate, Heavy, Severe)
        """
        try:
            # Check if this direction affects another intersection
            target_id = DIRECTIONAL_RELATIONSHIPS.get(self.intersection_id, {}).get(direction)
            if not target_id:
                logger.info(f"No target intersection for {direction} direction, not sending alert")
                return False
            
            # Create the alert message
            alert = {
                "timestamp": time.time(),
                "intersection_id": self.intersection_id,
                "intersection_name": self.intersection_name,
                "direction": direction,
                "congestion_level": congestion_level
            }
            
            # Publish to the congestion alerts topic
            topic = f"{CONGESTION_TOPIC}/{self.intersection_id}/{direction}"
            payload = json.dumps(alert)
            logger.info(f"Publishing congestion alert to {topic}: {congestion_level}")
            self.mqtt_client.publish(topic, payload, 1)
            
            # Track the alert
            alert_key = f"{self.intersection_id}_{direction}"
            self.congestion_alerts[alert_key] = {
                "direction": direction,
                "congestion_level": congestion_level,
                "timestamp": time.time(),
                "target_id": target_id
            }
            
            return True
        except Exception as e:
            logger.error(f"Error publishing congestion alert: {e}")
            return False
    
    def fetch_traffic_data(self):
        """
        Fetch and process traffic data from TomTom API
        
        Returns:
            dict: Processed traffic data by approach direction
        """
        try:
            # Get coordinates for this intersection's approaches
            approaches = self.intersection_info["approaches"]
            speed_limit_mph = self.intersection_info["speed_limit"]
            
            # Fetch traffic data for each approach
            approach_data = {}
            
            for direction, coords in approaches.items():
                lat, lon = coords
                flow_data = self.tomtom_client.get_traffic_flow(lat, lon)
                
                if flow_data and "flowSegmentData" in flow_data:
                    segment_data = flow_data["flowSegmentData"]
                    
                    # Extract key metrics
                    current_speed_kph = segment_data.get("currentSpeed", 0)
                    free_flow_speed_kph = segment_data.get("freeFlowSpeed", 0)
                    current_tt = segment_data.get("currentTravelTime", 0)
                    free_flow_tt = segment_data.get("freeFlowTravelTime", 0)
                    
                    # Convert speeds to mph for Georgia standards
                    current_speed_mph = current_speed_kph * 0.621371
                    free_flow_speed_mph = min(free_flow_speed_kph * 0.621371, speed_limit_mph)
                    
                    # Calculate metrics
                    speed_ratio = current_speed_mph / free_flow_speed_mph if free_flow_speed_mph > 0 else 0
                    delay = max(0, current_tt - free_flow_tt) if current_tt and free_flow_tt else 0
                    
                    # Determine congestion level
                    if speed_ratio >= 0.9:
                        congestion_level = "Free Flow"
                    elif speed_ratio >= 0.75:
                        congestion_level = "Light"
                    elif speed_ratio >= 0.5:
                        congestion_level = "Moderate"
                    elif speed_ratio >= 0.25:
                        congestion_level = "Heavy"
                    else:
                        congestion_level = "Severe"
                    
                    # Store processed data
                    approach_data[direction] = {
                        "current_speed_mph": round(current_speed_mph, 1),
                        "free_flow_speed_mph": round(free_flow_speed_mph, 1),
                        "speed_ratio": round(speed_ratio, 2),
                        "delay_seconds": delay,
                        "congestion_level": congestion_level
                    }
                    
                    # Check for significant congestion that should trigger an alert
                    if congestion_level in ["Moderate", "Heavy", "Severe"]:
                        # If this approach affects another intersection, send alert
                        if direction in DIRECTIONAL_RELATIONSHIPS.get(self.intersection_id, {}):
                            self._publish_congestion_alert(direction, congestion_level)
                            
            # Store traffic data for logging
            self.traffic_conditions = {
                "timestamp": time.time(),
                "approaches": approach_data
            }
            
            # Add to data log
            self.data_log.append(self.traffic_conditions)
            
            # Periodically save the log
            if len(self.data_log) % 10 == 0:
                self.save_data_log()
                
            return approach_data
            
        except Exception as e:
            logger.error(f"Error fetching traffic data: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def calculate_timing_plan(self, traffic_data):
        """
        Calculate optimal signal timing based on traffic data
        
        Args:
            traffic_data (dict): Processed traffic data by approach
            
        Returns:
            dict: Timing plan with green times for each phase
        """
        # If no data, use default timing
        if not traffic_data:
            logger.warning("Using default timing due to missing traffic data")
            return {
                Phase.NORTH_SOUTH: 30,
                Phase.EAST_WEST: 30
            }
        
        # Calculate congestion levels for each axis
        ns_congestion = []
        ew_congestion = []
        
        congestion_values = {
            "Free Flow": 0,
            "Light": 1,
            "Moderate": 2,
            "Heavy": 3,
            "Severe": 4
        }
        
        for direction, data in traffic_data.items():
            congestion_level = data.get("congestion_level")
            congestion_value = congestion_values.get(congestion_level, 0)
            
            if direction in ["North", "South"]:
                ns_congestion.append(congestion_value)
            elif direction in ["East", "West"]:
                ew_congestion.append(congestion_value)
        
        # Calculate average congestion for each axis
        ns_avg = sum(ns_congestion) / len(ns_congestion) if ns_congestion else 0
        ew_avg = sum(ew_congestion) / len(ew_congestion) if ew_congestion else 0
        
        # Calculate proportional green times
        total_congestion = ns_avg + ew_avg
        if total_congestion > 0:
            # More congested direction gets more green time
            ns_proportion = ns_avg / total_congestion
            ew_proportion = ew_avg / total_congestion
        else:
            # Equal split if no congestion
            ns_proportion = 0.5
            ew_proportion = 0.5
        
        # Define min and max green times
        min_green = 15
        max_green = 60
        available_green = 90  # Total available green time to distribute
        
        # Calculate green times based on congestion proportions
        ns_green = int(min_green + (available_green - 2 * min_green) * ns_proportion)
        ew_green = int(min_green + (available_green - 2 * min_green) * ew_proportion)
        
        # Ensure times are within bounds
        ns_green = max(min_green, min(max_green, ns_green))
        ew_green = max(min_green, min(max_green, ew_green))
        
        # Create timing plan
        timing_plan = {
            Phase.NORTH_SOUTH: ns_green,
            Phase.EAST_WEST: ew_green
        }
        
        logger.info(f"Calculated timing plan: NS={ns_green}s (congestion {ns_avg:.1f}), " 
                   f"EW={ew_green}s (congestion {ew_avg:.1f})")
        
        return timing_plan
    
    def update_timing(self):
        """Update signal timing based on latest traffic data"""
        # Check if it's time to update
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        
        logger.info(f"Time to update timing (role: {self.role})")
        
        # Fetch traffic data
        traffic_data = self.fetch_traffic_data()
        if not traffic_data:
            logger.warning("Failed to update timing due to missing traffic data")
            return
        
        # Calculate new timing plan
        timing_plan = self.calculate_timing_plan(traffic_data)
        
        # Update the signal controller
        self.signal_controller.update_timing_plan(timing_plan)
        
        # If master, publish timing plan to responders
        if self.role == "master":
            self._publish_timing_plan(timing_plan)
        
        # Publish status update
        self._publish_status("updated", {
            "traffic_conditions": {direction: data["congestion_level"] for direction, data in traffic_data.items()},
            "timing_plan": {phase.name: time for phase, time in timing_plan.items()}
        })
        
        # Update last update time
        self.last_update_time = current_time
        logger.info("Successfully updated signal timing")
    
    def save_data_log(self):
        """Save the data log to a file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.data_log, f, indent=2, default=str)
            logger.info(f"Saved data log to {self.log_file}")
        except Exception as e:
            logger.error(f"Error saving data log: {e}")
    
    def start(self):
        """Start the intersection coordinator"""
        try:
            logger.info(f"Connecting to AWS IoT endpoint: {ENDPOINT}")
            connect_result = self.mqtt_client.connect()
            
            if not connect_result:
                logger.error("Failed to connect to AWS IoT")
                return
                
            logger.info("Connected to AWS IoT")
            self._publish_status("starting")
            
            # Start the signal controller
            self.signal_controller.start_control_loop()
            logger.info("Signal control loop started")
            
            # Initial traffic data fetch and timing update
            traffic_data = self.fetch_traffic_data()
            if traffic_data:
                timing_plan = self.calculate_timing_plan(traffic_data)
                self.signal_controller.update_timing_plan(timing_plan)
                
                # If master, publish initial timing plan
                if self.role == "master":
                    self._publish_timing_plan(timing_plan)
            
            # Publish running status
            self._publish_status("running")
            
            # Main control loop
            logger.info(f"Starting main control loop with {self.update_interval}s update interval")
            self.last_update_time = time.time()
            self.last_status_time = time.time()
            
            while True:
                # Update timing periodically
                self.update_timing()
                
                # Publish regular status updates
                current_time = time.time()
                if current_time - self.last_status_time >= 60:  # Every minute
                    self._publish_status("heartbeat", {
                        "uptime": int(current_time - self.last_update_time),
                        "congestion_alerts": len(self.congestion_alerts),
                        "timing_adjustments": len(self.timing_adjustments)
                    })
                    self.last_status_time = current_time
                
                # Sleep to prevent high CPU usage
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("Intersection coordinator stopped by user")
        except Exception as e:
            logger.error(f"Error in intersection coordinator: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        # Save any remaining data
        self.save_data_log()
        
        # Publish shutdown status
        self._publish_status("shutting_down")
        
        # Disconnect from MQTT
        try:
            self.mqtt_client.disconnect()
            logger.info("Disconnected from AWS IoT")
        except Exception as e:
            logger.error(f"Error disconnecting from AWS IoT: {e}")
        
        # Clean up the signal controller
        if hasattr(self, 'signal_controller'):
            self.signal_controller.cleanup()
        
        logger.info("Intersection coordinator cleaned up")


# Main entry point
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Start an intersection coordinator')
    parser.add_argument('--intersection', '-i', type=str, required=True,
                      choices=list(INTERSECTIONS.keys()),
                      help='Name of the intersection to control')
    parser.add_argument('--role', '-r', type=str, default="master",
                      choices=["master", "responder"],
                      help='Role of this coordinator (master or responder)')
    parser.add_argument('--interval', '-t', type=int, default=60,
                      help='Update interval in seconds (default: 60)')
    parser.add_argument('--mock', '-m', action='store_true',
                      help='Use mock signal controller instead of hardware')
    
    args = parser.parse_args()
    
    # Print configuration
    print(f"Starting coordinator for {args.intersection} in {args.role} role")
    print(f"Update interval: {args.interval} seconds")
    print(f"Using mock controller: {args.mock}")
    
    # Create and start coordinator
    coordinator = IntersectionCoordinator(
        intersection_name=args.intersection,
        role=args.role,
        update_interval=args.interval,
        use_mock=args.mock
    )
    
    coordinator.start()