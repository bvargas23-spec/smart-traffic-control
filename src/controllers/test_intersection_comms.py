#!/usr/bin/env python3
# test_intersection_comms.py - Test communication between intersections
import json
import time
import threading
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import os
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TestIntersectionComms")

# AWS IoT Configuration
ENDPOINT = "a2ao1owrs8g0lu-ats.iot.us-east-2.amazonaws.com"
CERT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs/certificate.pem.crt")
PRIVATE_KEY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs/private.pem.key")
ROOT_CA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs/AmazonRootCA1.pem")

# MQTT Topics
CONGESTION_TOPIC = "traffic/alerts"  # Base topic for congestion alerts
STATUS_TOPIC = "traffic/intersection/status"
TIMING_TOPIC = "traffic/intersection/timing"
TEST_TOPIC = "traffic/test"

# Intersection configuration
INTERSECTIONS = {
    "roswell_hwy41": {
        "name": "Roswell & Hwy 41",
        "directions": ["North", "South", "East", "West"]
    },
    "cobb_hwy120": {
        "name": "Cobb Pkwy & Hwy 120",
        "directions": ["North", "South", "East", "West"]
    }
}

# Test messages for different scenarios
TEST_MESSAGES = {
    "congestion_alert": {
        "timestamp": None,  # Will be populated at runtime
        "intersection_id": None,  # Will be populated based on sender
        "intersection_name": None,  # Will be populated based on sender
        "direction": None,  # Will be populated based on direction
        "congestion_level": None  # Will be populated based on level
    },
    "timing_update": {
        "timestamp": None,  # Will be populated at runtime
        "intersection_id": None,  # Will be populated based on sender
        "intersection_name": None,  # Will be populated based on sender
        "timing_plan": {
            "NORTH_SOUTH": 30,
            "EAST_WEST": 30
        }
    },
    "status_update": {
        "timestamp": None,  # Will be populated at runtime
        "intersection_id": None,  # Will be populated based on sender
        "intersection_name": None,  # Will be populated based on sender
        "role": "master",
        "status": "running"
    }
}

class IntersectionCommunicationTester:
    """Test communication between intersections via AWS IoT"""
    
    def __init__(self, client_id="TestClient"):
        self.client_id = client_id
        self.received_messages = []
        self.mqtt_client = self._setup_mqtt_client()
        self.connection_established = False
        self.lock = threading.Lock()
    
    def _setup_mqtt_client(self):
        """Set up the MQTT client"""
        mqtt_client = AWSIoTMQTTClient(self.client_id)
        mqtt_client.configureEndpoint(ENDPOINT, 8883)
        mqtt_client.configureCredentials(ROOT_CA_PATH, PRIVATE_KEY_PATH, CERT_PATH)
        mqtt_client.configureAutoReconnectBackoffTime(1, 32, 20)
        mqtt_client.configureOfflinePublishQueueing(-1)  # Infinite offline queuing
        mqtt_client.configureDrainingFrequency(2)  # Draining: 2 Hz
        mqtt_client.configureConnectDisconnectTimeout(10)  # 10 seconds
        mqtt_client.configureMQTTOperationTimeout(5)  # 5 seconds
        
        # Setup callbacks
        def on_online_callback():
            logger.info("MQTT Connection established - client is online")
            self.connection_established = True
            
        mqtt_client.onOnline = on_online_callback
        
        def on_offline_callback():
            logger.warning("MQTT Connection lost - client is offline")
            self.connection_established = False
            
        mqtt_client.onOffline = on_offline_callback
        
        return mqtt_client
    
    def start(self):
        """Start the tester"""
        logger.info(f"Connecting to AWS IoT endpoint: {ENDPOINT}")
        
        try:
            # Connect to AWS IoT
            connect_result = self.mqtt_client.connect()
            if not connect_result:
                logger.error("Failed to connect to AWS IoT")
                return False
            
            logger.info(f"Connected to AWS IoT as {self.client_id}")
            
            # Wait for the connection to be fully established
            for _ in range(10):
                if self.connection_established:
                    break
                time.sleep(0.5)
            
            if not self.connection_established:
                logger.warning("Connection callback not triggered, but proceeding anyway")
            
            # Subscribe to all relevant topics
            self._subscribe_to_topics()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting tester: {e}")
            return False
    
    def _subscribe_to_topics(self):
        """Subscribe to all relevant topics"""
        # Subscribe to congestion alerts
        self.mqtt_client.subscribe(f"{CONGESTION_TOPIC}/#", 1, self._message_callback)
        logger.info(f"Subscribed to {CONGESTION_TOPIC}/#")
        
        # Subscribe to status updates
        self.mqtt_client.subscribe(f"{STATUS_TOPIC}/#", 1, self._message_callback)
        logger.info(f"Subscribed to {STATUS_TOPIC}/#")
        
        # Subscribe to timing updates
        self.mqtt_client.subscribe(f"{TIMING_TOPIC}/#", 1, self._message_callback)
        logger.info(f"Subscribed to {TIMING_TOPIC}/#")
        
        # Subscribe to test topic
        self.mqtt_client.subscribe(f"{TEST_TOPIC}/#", 1, self._message_callback)
        logger.info(f"Subscribed to {TEST_TOPIC}/#")
    
    def _message_callback(self, _client, _userdata, message):
        """Callback for received messages"""
        try:
            # Parse the message payload
            try:
                payload = json.loads(message.payload)
            except json.JSONDecodeError:
                payload = {"raw": message.payload.decode('utf-8')}
            
            # Create message record
            received_msg = {
                "topic": message.topic,
                "payload": payload,
                "timestamp": time.time()
            }
            
            # Log the message
            logger.info(f"Received message on {message.topic}")
            
            # Store the message
            with self.lock:
                self.received_messages.append(received_msg)
                
        except Exception as e:
            logger.error(f"Error in message callback: {e}")
    
    def send_test_congestion_alert(self, intersection_id, direction, level):
        """
        Send a test congestion alert
        
        Args:
            intersection_id (str): ID of the intersection sending the alert
            direction (str): Direction of congestion (North, South, East, West)
            level (str): Congestion level (Free Flow, Light, Moderate, Heavy, Severe)
        """
        # Get intersection details
        if intersection_id not in INTERSECTIONS:
            logger.error(f"Unknown intersection ID: {intersection_id}")
            return False
        
        intersection = INTERSECTIONS[intersection_id]
        
        # Verify direction
        if direction not in intersection["directions"]:
            logger.error(f"Invalid direction '{direction}' for intersection {intersection_id}")
            return False
        
        # Verify congestion level
        valid_levels = ["Free Flow", "Light", "Moderate", "Heavy", "Severe"]
        if level not in valid_levels:
            logger.error(f"Invalid congestion level: {level}")
            return False
        
        # Create the message
        message = TEST_MESSAGES["congestion_alert"].copy()
        message["timestamp"] = time.time()
        message["intersection_id"] = intersection_id
        message["intersection_name"] = intersection["name"]
        message["direction"] = direction
        message["congestion_level"] = level
        
        # Publish the message
        topic = f"{CONGESTION_TOPIC}/{intersection_id}/{direction}"
        logger.info(f"Publishing congestion alert to {topic}: {level}")
        result = self.mqtt_client.publish(topic, json.dumps(message), 1)
        
        return result
    
    def send_test_timing_update(self, intersection_id, north_south_time, east_west_time):
        """
        Send a test timing update
        
        Args:
            intersection_id (str): ID of the intersection
            north_south_time (int): Green time for north-south phase
            east_west_time (int): Green time for east-west phase
        """
        # Get intersection details
        if intersection_id not in INTERSECTIONS:
            logger.error(f"Unknown intersection ID: {intersection_id}")
            return False
        
        intersection = INTERSECTIONS[intersection_id]
        
        # Create the message
        message = TEST_MESSAGES["timing_update"].copy()
        message["timestamp"] = time.time()
        message["intersection_id"] = intersection_id
        message["intersection_name"] = intersection["name"]
        message["timing_plan"]["NORTH_SOUTH"] = north_south_time
        message["timing_plan"]["EAST_WEST"] = east_west_time
        
        # Publish the message
        topic = f"{TIMING_TOPIC}/{intersection_id}"
        logger.info(f"Publishing timing update to {topic}")
        result = self.mqtt_client.publish(topic, json.dumps(message), 1)
        
        return result
    
    def send_test_status_update(self, intersection_id, status, role="master"):
        """
        Send a test status update
        
        Args:
            intersection_id (str): ID of the intersection
            status (str): Status of the intersection
            role (str): Role of the intersection (master or responder)
        """
        # Get intersection details
        if intersection_id not in INTERSECTIONS:
            logger.error(f"Unknown intersection ID: {intersection_id}")
            return False
        
        intersection = INTERSECTIONS[intersection_id]
        
        # Create the message
        message = TEST_MESSAGES["status_update"].copy()
        message["timestamp"] = time.time()
        message["intersection_id"] = intersection_id
        message["intersection_name"] = intersection["name"]
        message["status"] = status
        message["role"] = role
        
        # Publish the message
        topic = f"{STATUS_TOPIC}/{intersection_id}"
        logger.info(f"Publishing status update to {topic}: {status}")
        result = self.mqtt_client.publish(topic, json.dumps(message), 1)
        
        return result
    
    def wait_for_messages(self, topic_filter=None, count=1, timeout=10):
        """
        Wait for a specified number of messages to be received
        
        Args:
            topic_filter (str): Optional topic filter
            count (int): Number of messages to wait for
            timeout (int): Timeout in seconds
            
        Returns:
            list: Received messages
        """
        start_time = time.time()
        matched_messages = []
        
        while time.time() - start_time < timeout:
            with self.lock:
                # Filter messages if a topic filter was provided
                if topic_filter:
                    messages = [msg for msg in self.received_messages if msg["topic"].startswith(topic_filter)]
                else:
                    messages = self.received_messages.copy()
                    
                # Take what we need and remove from the main list
                new_matches = messages[:count - len(matched_messages)]
                matched_messages.extend(new_matches)
                
                # If we have enough messages, return them
                if len(matched_messages) >= count:
                    # Remove these messages from the main list
                    for match in new_matches:
                        if match in self.received_messages:
                            self.received_messages.remove(match)
                    return matched_messages
            
            # Not enough messages yet, wait a bit
            time.sleep(0.1)
        
        # Timeout reached, return what we have
        return matched_messages
    
    def clear_messages(self):
        """Clear all received messages"""
        with self.lock:
            self.received_messages.clear()
    
    def disconnect(self):
        """Disconnect from AWS IoT"""
        try:
            self.mqtt_client.disconnect()
            logger.info("Disconnected from AWS IoT")
        except Exception as e:
            logger.error(f"Error disconnecting from AWS IoT: {e}")


def test_intersections_congestion_alert():
    """Test that a congestion alert from one intersection triggers a response"""
    # Create tester
    tester = IntersectionCommunicationTester(client_id="CongestionTester")
    
    # Start tester
    if not tester.start():
        logger.error("Failed to start tester")
        return
    
    try:
        # Clear any existing messages
        tester.clear_messages()
        
        # Send a congestion alert from Roswell to Cobb (South direction connects them)
        logger.info("\n=== Testing Roswell -> Cobb Congestion Alert (South) ===")
        tester.send_test_congestion_alert("roswell_hwy41", "South", "Heavy")
        
        # Wait for a response (status update from Cobb)
        response = tester.wait_for_messages(topic_filter=f"{STATUS_TOPIC}/cobb_hwy120", timeout=5)
        
        if response:
            logger.info(f"✅ Received response from Cobb: {response[0]['payload'].get('status')}")
            
            # Check if it mentions an adjustment
            if "adjusted" in response[0]['payload'].get('status', ''):
                logger.info("✅ Cobb correctly adjusted its timing in response to the alert")
            else:
                logger.warning("⚠️ Cobb did not adjust its timing in response to the alert")
        else:
            logger.warning("⚠️ No response received from Cobb")
        
        # Now test in the other direction
        logger.info("\n=== Testing Cobb -> Roswell Congestion Alert (North) ===")
        tester.clear_messages()
        tester.send_test_congestion_alert("cobb_hwy120", "North", "Severe")
        
        # Wait for a response (status update from Roswell)
        response = tester.wait_for_messages(topic_filter=f"{STATUS_TOPIC}/roswell_hwy41", timeout=5)
        
        if response:
            logger.info(f"✅ Received response from Roswell: {response[0]['payload'].get('status')}")
            
            # Check if it mentions an adjustment
            if "adjusted" in response[0]['payload'].get('status', ''):
                logger.info("✅ Roswell correctly adjusted its timing in response to the alert")
            else:
                logger.warning("⚠️ Roswell did not adjust its timing in response to the alert")
        else:
            logger.warning("⚠️ No response received from Roswell")
        
        # Test the East/West axes - there should be no response since they aren't connected
        logger.info("\n=== Testing Non-Connected Direction (East) ===")
        tester.clear_messages()
        tester.send_test_congestion_alert("roswell_hwy41", "East", "Severe")
        
        # Brief wait to see if there's a response (there shouldn't be)
        response = tester.wait_for_messages(topic_filter=f"{STATUS_TOPIC}/", timeout=3)
        
        if not response:
            logger.info("✅ No response to East congestion as expected (not connected)")
        else:
            logger.warning(f"⚠️ Unexpected response to East congestion: {response}")
        
    finally:
        # Disconnect
        tester.disconnect()


def test_timing_command_propagation():
    """Test that a timing command from a master is received by a responder"""
    # Create tester
    tester = IntersectionCommunicationTester(client_id="TimingTester")
    
    # Start tester
    if not tester.start():
        logger.error("Failed to start tester")
        return
    
    try:
        # Clear any existing messages
        tester.clear_messages()
        
        # Send a timing update from Roswell (acting as master)
        logger.info("\n=== Testing Timing Command Propagation ===")
        tester.send_test_timing_update("roswell_hwy41", 45, 25)  # NS=45s, EW=25s
        
        # Wait for a response (status update from Cobb acknowledging timing update)
        response = tester.wait_for_messages(topic_filter=f"{STATUS_TOPIC}/cobb_hwy120", timeout=5)
        
        if response:
            logger.info(f"✅ Received response from Cobb: {response[0]['payload'].get('status')}")
            
            # Check if it mentions an update
            if "updated" in response[0]['payload'].get('status', ''):
                logger.info("✅ Cobb correctly acknowledged timing update")
            else:
                logger.warning("⚠️ Cobb did not acknowledge timing update")
        else:
            logger.warning("⚠️ No response received from Cobb")
        
    finally:
        # Disconnect
        tester.disconnect()


def run_full_communication_test():
    """Run a comprehensive test of all communication functions"""
    # Create tester
    tester = IntersectionCommunicationTester(client_id="FullTester")
    
    # Start tester
    if not tester.start():
        logger.error("Failed to start tester")
        return
    
    try:
        # Wait for both intersections to report their status
        logger.info("\n=== Waiting for both intersections to come online ===")
        
        # Wait for status messages from both intersections
        roswell_status = tester.wait_for_messages(topic_filter=f"{STATUS_TOPIC}/roswell_hwy41", timeout=30)
        cobb_status = tester.wait_for_messages(topic_filter=f"{STATUS_TOPIC}/cobb_hwy120", timeout=5)
        
        if roswell_status:
            logger.info(f"✅ Roswell is online: {roswell_status[0]['payload'].get('status')}")
        else:
            logger.warning("⚠️ Roswell is not online")
            
        if cobb_status:
            logger.info(f"✅ Cobb is online: {cobb_status[0]['payload'].get('status')}")
        else:
            logger.warning("⚠️ Cobb is not online")
        
        # Clear messages
        tester.clear_messages()
        
        # Test sequence 1: Moderate congestion from Roswell to Cobb
        logger.info("\n=== Test 1: Moderate congestion from Roswell to Cobb ===")
        tester.send_test_congestion_alert("roswell_hwy41", "South", "Moderate")
        time.sleep(5)  # Wait for processing
        
        # Test sequence 2: Heavy congestion from Cobb to Roswell
        logger.info("\n=== Test 2: Heavy congestion from Cobb to Roswell ===")
        tester.send_test_congestion_alert("cobb_hwy120", "North", "Heavy")
        time.sleep(5)  # Wait for processing
        
        # Test sequence 3: Severe congestion both ways
        logger.info("\n=== Test 3: Severe congestion both ways ===")
        tester.send_test_congestion_alert("roswell_hwy41", "South", "Severe")
        tester.send_test_congestion_alert("cobb_hwy120", "North", "Severe")
        time.sleep(5)  # Wait for processing
        
        # Get all received messages
        all_messages = tester.wait_for_messages(timeout=1)  # Just to get any remaining messages
        
        # Report summary
        congestion_alerts = [msg for msg in all_messages if CONGESTION_TOPIC in msg["topic"]]
        status_updates = [msg for msg in all_messages if STATUS_TOPIC in msg["topic"]]
        timing_updates = [msg for msg in all_messages if TIMING_TOPIC in msg["topic"]]
        
        logger.info("\n=== Communication Test Summary ===")
        logger.info(f"Total messages: {len(all_messages)}")
        logger.info(f"Congestion alerts: {len(congestion_alerts)}")
        logger.info(f"Status updates: {len(status_updates)}")
        logger.info(f"Timing updates: {len(timing_updates)}")
        
        if timing_updates:
            logger.info("\nTiming adjustments detected:")
            for msg in timing_updates:
                try:
                    plan = msg["payload"].get("timing_plan", {})
                    sender = msg["payload"].get("intersection_id", "unknown")
                    logger.info(f"- {sender}: NS={plan.get('NORTH_SOUTH')}s, EW={plan.get('EAST_WEST')}s")
                except Exception:
                    pass
                    
        if status_updates:
            logger.info("\nStatus updates with adjustments:")
            for msg in status_updates:
                try:
                    status = msg["payload"].get("status", "")
                    sender = msg["payload"].get("intersection_id", "unknown")
                    if "adjusted" in status:
                        adjustment = msg["payload"].get("adjustment", "unknown")
                        direction = msg["payload"].get("direction", "unknown")
                        logger.info(f"- {sender}: {adjustment}s adjustment due to {direction} congestion")
                except Exception:
                    pass
        
    finally:
        # Disconnect
        tester.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test intersection communication')
    parser.add_argument('--test', choices=['congestion', 'timing', 'full'], default='full',
                      help='Test to run (default: full)')
    
    args = parser.parse_args()
    
    if args.test == 'congestion':
        test_intersections_congestion_alert()
    elif args.test == 'timing':
        test_timing_command_propagation()
    else:  # full
        run_full_communication_test()