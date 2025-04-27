#!/usr/bin/env python3
# dashboard.py - Simple web dashboard for monitoring traffic control system

import os
import json
import time
import threading
from datetime import datetime
import logging
from flask import Flask, render_template, jsonify, request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TrafficDashboard")

# AWS IoT client for receiving status updates
try:
    from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
    aws_iot_available = True
except ImportError:
    logger.warning("AWS IoT Python SDK not available, using simulated data only")
    aws_iot_available = False

# AWS IoT configuration
ENDPOINT = "a2ao1owrs8g0lu-ats.iot.us-east-2.amazonaws.com"
CERT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs/dashboard.pem.crt")
PRIVATE_KEY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs/dashboard.pem.key")
ROOT_CA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs/AmazonRootCA1.pem")

# MQTT Topics
STATUS_TOPIC = "traffic/intersection/#"
TIMING_TOPIC = "traffic/intersection/timing/#"
CONGESTION_TOPIC = "traffic/#"

# Debug options
ENABLE_MQTT_LOGGING = True

# Intersection definitions
INTERSECTIONS = {
    "roswell_hwy41": {
        "name": "Roswell & Hwy 41",
        "coordinates": {
            "latitude": 33.960192828395996,
            "longitude": -84.52790520126695
        },
        "approaches": ["North", "South", "East", "West"]
    },
    "cobb_hwy120": {
        "name": "Cobb Pkwy & Hwy 120",
        "coordinates": {
            "latitude": 33.94178,
            "longitude": -84.51568
        },
        "approaches": ["North", "South", "East", "West"]
    }
}

# Global state for dashboard
dashboard_state = {
    "intersections": {},
    "alerts": [],
    "last_update": None
}

# Initialize with empty data for each intersection
for intersection_id, info in INTERSECTIONS.items():
    dashboard_state["intersections"][intersection_id] = {
        "name": info["name"],
        "status": "unknown",
        "role": "unknown",
        "timing_plan": {
            "NORTH_SOUTH": 30,
            "EAST_WEST": 30
        },
        "traffic_conditions": {
            approach: "Unknown" for approach in info["approaches"]
        },
        "last_update": None
    }

# Maximum number of alerts to keep
MAX_ALERTS = 20

# Flask app
app = Flask(__name__)

# Setup AWS IoT client
mqtt_client = None

if aws_iot_available:
    def setup_mqtt_client():
        global mqtt_client
        
        client_id = f"dashboard_{int(time.time())}"
        mqtt_client = AWSIoTMQTTClient(client_id)
        mqtt_client.configureEndpoint(ENDPOINT, 8883)
        mqtt_client.configureCredentials(ROOT_CA_PATH, PRIVATE_KEY_PATH, CERT_PATH)
        mqtt_client.configureAutoReconnectBackoffTime(1, 32, 20)
        mqtt_client.configureOfflinePublishQueueing(-1)
        mqtt_client.configureDrainingFrequency(2)
        mqtt_client.configureConnectDisconnectTimeout(10)
        mqtt_client.configureMQTTOperationTimeout(5)
        
        # Set up callbacks for connection status
        def on_online_callback():
            logger.info("MQTT Connection established - dashboard is online")
            
        def on_offline_callback():
            logger.warning("MQTT Connection lost - dashboard is offline")
            
        mqtt_client.onOnline = on_online_callback
        mqtt_client.onOffline = on_offline_callback
        
        # Subscribe to topics
        mqtt_client.subscribe(STATUS_TOPIC, 1, status_callback)
        mqtt_client.subscribe(TIMING_TOPIC, 1, timing_callback)
        mqtt_client.subscribe(CONGESTION_TOPIC, 1, congestion_callback)
        
        # Connect to AWS IoT
        logger.info(f"Attempting to connect to AWS IoT at {ENDPOINT}")
        logger.info(f"Using certificates: {CERT_PATH}, {PRIVATE_KEY_PATH}, {ROOT_CA_PATH}")
        
        connect_result = mqtt_client.connect()
        if not connect_result:
            logger.error("Failed to connect to AWS IoT")
            return False
            
        logger.info("Connected to AWS IoT")
        return True

    def status_callback(client, userdata, message):
        """Callback for status messages"""
        try:
            topic = message.topic
            
            # Log the incoming message for debugging
            if ENABLE_MQTT_LOGGING:
                logger.info(f"Received message on topic: {topic}")
                logger.info(f"Payload: {message.payload}")
            
            # Skip timing messages (they'll be handled by timing_callback)
            if "timing" in topic:
                return
                
            payload = json.loads(message.payload)
            
            # Extract intersection ID from topic if not in payload
            if "intersection_id" not in payload and "intersection" in topic:
                parts = topic.split('/')
                if len(parts) >= 3:
                    # Try to find the intersection ID from the topic
                    intersection_part = None
                    for part in parts:
                        if part in INTERSECTIONS:
                            intersection_part = part
                            break
                    
                    if intersection_part:
                        payload["intersection_id"] = intersection_part
                        logger.info(f"Extracted intersection_id from topic: {intersection_part}")
            
            process_status_update(payload)
        except Exception as e:
            logger.error(f"Error processing status message: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def timing_callback(client, userdata, message):
        """Callback for timing messages"""
        try:
            topic = message.topic
            
            # Log the incoming message for debugging
            if ENABLE_MQTT_LOGGING:
                logger.info(f"Received timing message on topic: {topic}")
                logger.info(f"Timing payload: {message.payload}")
                
            payload = json.loads(message.payload)
            
            # Extract intersection ID from topic if not in payload
            if "intersection_id" not in payload and "intersection" in topic:
                parts = topic.split('/')
                for idx, part in enumerate(parts):
                    if part == "intersection" and idx + 1 < len(parts):
                        possible_id = parts[idx + 1]
                        if possible_id in INTERSECTIONS:
                            payload["intersection_id"] = possible_id
                            logger.info(f"Extracted intersection_id from timing topic: {possible_id}")
                            break
            
            # Look for timing plan data in different formats
            if "timing_plan" not in payload:
                # Try alternative field names
                for field in ["timing", "signal_timings", "plan"]:
                    if field in payload:
                        payload["timing_plan"] = payload[field]
                        logger.info(f"Found timing data in field: {field}")
                        break
                        
                # If still not found, check if the payload itself is the timing plan
                if "timing_plan" not in payload and ("NORTH_SOUTH" in payload or "north_south" in payload):
                    # The payload itself might be the timing plan
                    logger.info("Payload appears to be the timing plan itself")
                    
                    # Normalize field names
                    timing_plan = {}
                    if "NORTH_SOUTH" in payload:
                        timing_plan["NORTH_SOUTH"] = payload["NORTH_SOUTH"]
                    elif "north_south" in payload:
                        timing_plan["NORTH_SOUTH"] = payload["north_south"]
                        
                    if "EAST_WEST" in payload:
                        timing_plan["EAST_WEST"] = payload["EAST_WEST"]
                    elif "east_west" in payload:
                        timing_plan["EAST_WEST"] = payload["east_west"]
                        
                    payload["timing_plan"] = timing_plan
            
            process_timing_update(payload)
        except Exception as e:
            logger.error(f"Error processing timing message: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def congestion_callback(client, userdata, message):
        """Callback for congestion alerts"""
        try:
            topic = message.topic
            
            # Log the incoming message for debugging
            if ENABLE_MQTT_LOGGING:
                logger.info(f"Received congestion message on topic: {topic}")
                logger.info(f"Congestion payload: {message.payload}")
                
            payload = json.loads(message.payload)
            
            # Skip non-congestion messages
            if not any(key in topic for key in ["alerts", "congestion", "traffic"]):
                return
                
            # Extract intersection ID from topic if not in payload
            if "intersection_id" not in payload and "intersection" in topic:
                parts = topic.split('/')
                for idx, part in enumerate(parts):
                    if part == "intersection" and idx + 1 < len(parts):
                        possible_id = parts[idx + 1]
                        if possible_id in INTERSECTIONS:
                            payload["intersection_id"] = possible_id
                            logger.info(f"Extracted intersection_id from congestion topic: {possible_id}")
                            break
            
            # Look for congestion data in different formats
            if "direction" not in payload or "congestion_level" not in payload:
                # Check if this is a status update with traffic conditions
                if "traffic_conditions" in payload and "intersection_id" in payload:
                    # This is a status update with traffic conditions
                    # Process it as multiple congestion alerts
                    for direction, level in payload["traffic_conditions"].items():
                        alert_payload = {
                            "intersection_id": payload["intersection_id"],
                            "direction": direction,
                            "congestion_level": level
                        }
                        process_congestion_alert(alert_payload)
                    return
            
            process_congestion_alert(payload)
        except Exception as e:
            logger.error(f"Error processing congestion message: {e}")
            import traceback
            logger.error(traceback.format_exc())

def process_status_update(payload):
    """Process a status update message"""
    global dashboard_state
    
    intersection_id = payload.get("intersection_id")
    if not intersection_id or intersection_id not in dashboard_state["intersections"]:
        logger.warning(f"Unknown intersection_id in status update: {intersection_id}")
        return
        
    # Update intersection data
    intersection = dashboard_state["intersections"][intersection_id]
    if "status" in payload:
        intersection["status"] = payload.get("status", "unknown")
    if "role" in payload:
        intersection["role"] = payload.get("role", "unknown")
    intersection["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check for traffic conditions in the update
    if "traffic_conditions" in payload:
        for direction, level in payload["traffic_conditions"].items():
            if direction in intersection["traffic_conditions"]:
                intersection["traffic_conditions"][direction] = level
    
    # Handle specific fields from the payload
    for field in ["north_south", "east_west", "NORTH_SOUTH", "EAST_WEST"]:
        if field in payload:
            # This might be a timing update in a status message
            normalized_field = field.upper()
            intersection["timing_plan"][normalized_field] = payload[field]
    
    # Update dashboard last update time
    dashboard_state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info(f"Updated status for {intersection_id}: {intersection['status']}")

def process_timing_update(payload):
    """Process a timing update message"""
    global dashboard_state
    
    intersection_id = payload.get("intersection_id")
    if not intersection_id or intersection_id not in dashboard_state["intersections"]:
        logger.warning(f"Unknown intersection_id in timing update: {intersection_id}")
        return
        
    # Update timing plan
    if "timing_plan" in payload:
        timing_plan = payload["timing_plan"]
        updated_plan = {}
        
        # Try to extract timing values in different formats
        for field in ["NORTH_SOUTH", "north_south", "ns"]:
            if field in timing_plan:
                val = timing_plan[field]
                # Convert to int if it's a string
                if isinstance(val, str) and val.isdigit():
                    val = int(val)
                updated_plan["NORTH_SOUTH"] = val
                break
                
        for field in ["EAST_WEST", "east_west", "ew"]:
            if field in timing_plan:
                val = timing_plan[field]
                # Convert to int if it's a string
                if isinstance(val, str) and val.isdigit():
                    val = int(val)
                updated_plan["EAST_WEST"] = val
                break
        
        # Update only if we found valid timing data
        if updated_plan:
            dashboard_state["intersections"][intersection_id]["timing_plan"].update(updated_plan)
            
            # Add to alerts
            ns_time = updated_plan.get("NORTH_SOUTH", 
                       dashboard_state["intersections"][intersection_id]["timing_plan"].get("NORTH_SOUTH", "unknown"))
            ew_time = updated_plan.get("EAST_WEST", 
                       dashboard_state["intersections"][intersection_id]["timing_plan"].get("EAST_WEST", "unknown"))
            
            alert = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "timing",
                "intersection": dashboard_state["intersections"][intersection_id]["name"],
                "message": f"Timing updated: NS={ns_time}s, EW={ew_time}s"
            }
            dashboard_state["alerts"].insert(0, alert)
            
            # Trim alerts list if needed
            if len(dashboard_state["alerts"]) > MAX_ALERTS:
                dashboard_state["alerts"] = dashboard_state["alerts"][:MAX_ALERTS]
            
            # Update dashboard last update time
            dashboard_state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"Updated timing for {intersection_id}: NS={ns_time}s, EW={ew_time}s")

def process_congestion_alert(payload):
    """Process a congestion alert message"""
    global dashboard_state
    
    intersection_id = payload.get("intersection_id")
    direction = payload.get("direction")
    congestion_level = payload.get("congestion_level")
    
    if not all([intersection_id, direction, congestion_level]):
        logger.warning(f"Missing required fields in congestion alert: {payload}")
        return
        
    if intersection_id not in dashboard_state["intersections"]:
        logger.warning(f"Unknown intersection_id in congestion alert: {intersection_id}")
        return
        
    # Add to alerts
    alert = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "congestion",
        "intersection": dashboard_state["intersections"][intersection_id]["name"],
        "message": f"{direction} approach has {congestion_level} congestion"
    }
    dashboard_state["alerts"].insert(0, alert)
    
    # Update traffic conditions for this approach
    if direction in dashboard_state["intersections"][intersection_id]["traffic_conditions"]:
        dashboard_state["intersections"][intersection_id]["traffic_conditions"][direction] = congestion_level
    
    # Trim alerts list if needed
    if len(dashboard_state["alerts"]) > MAX_ALERTS:
        dashboard_state["alerts"] = dashboard_state["alerts"][:MAX_ALERTS]
    
    # Update dashboard last update time
    dashboard_state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info(f"Added congestion alert for {intersection_id} {direction}: {congestion_level}")

def generate_simulated_data():
    """Generate simulated data for testing"""
    global dashboard_state
    
    congestion_levels = ["Free Flow", "Light", "Moderate", "Heavy", "Severe"]
    statuses = ["running", "adjusted", "updated"]
    
    for intersection_id, intersection in dashboard_state["intersections"].items():
        # Simulate status changes
        intersection["status"] = statuses[int(time.time() / 10) % len(statuses)]
        
        # Simulate timing changes
        ns_time = 30 + ((int(time.time() / 30) % 7) * 5)  # 30-60 seconds
        ew_time = 90 - ns_time  # Complementary to NS time
        intersection["timing_plan"] = {
            "NORTH_SOUTH": ns_time,
            "EAST_WEST": ew_time
        }
        
        # Simulate traffic conditions
        for approach in intersection["traffic_conditions"]:
            idx = (int(time.time() / 20) + ord(approach[0])) % len(congestion_levels)
            intersection["traffic_conditions"][approach] = congestion_levels[idx]
            
        intersection["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Simulate alerts
    if len(dashboard_state["alerts"]) < MAX_ALERTS and int(time.time()) % 10 == 0:
        intersection_id = list(dashboard_state["intersections"].keys())[int(time.time() / 5) % len(dashboard_state["intersections"])]
        intersection_name = dashboard_state["intersections"][intersection_id]["name"]
        direction = INTERSECTIONS[intersection_id]["approaches"][int(time.time() / 7) % len(INTERSECTIONS[intersection_id]["approaches"])]
        congestion = congestion_levels[int(time.time() / 11) % len(congestion_levels)]
        
        alert = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "congestion",
            "intersection": intersection_name,
            "message": f"{direction} approach has {congestion} congestion"
        }
        dashboard_state["alerts"].insert(0, alert)
    
    # Update dashboard last update time
    dashboard_state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def simulation_thread():
    """Thread function for generating simulated data"""
    while True:
        generate_simulated_data()
        time.sleep(5)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dashboard-state')
def get_dashboard_state():
    return jsonify(dashboard_state)

@app.route('/api/clear-alerts', methods=['POST'])
def clear_alerts():
    global dashboard_state
    dashboard_state["alerts"] = []
    return jsonify({"status": "success"})

@app.route('/api/connection-status')
def connection_status():
    """Check if the dashboard is connected to AWS IoT"""
    if not aws_iot_available:
        return jsonify({"status": "unavailable", "message": "AWS IoT Python SDK not available"})
        
    if mqtt_client is None:
        return jsonify({"status": "not_initialized", "message": "MQTT client not initialized"})
        
    # Check if the client is connected
    if mqtt_client._mqtt_core.getClient()._mqtt_client.is_connected():
        return jsonify({"status": "connected", "message": "Connected to AWS IoT"})
    else:
        return jsonify({"status": "disconnected", "message": "Not connected to AWS IoT"})

def start_dashboard(host='0.0.0.0', port=8080, debug=False, use_simulated_data=True):
    # Start MQTT client in background thread if available
    if aws_iot_available:
        mqtt_thread = threading.Thread(target=setup_mqtt_client)
        mqtt_thread.daemon = True
        mqtt_thread.start()
        # Give the MQTT client a chance to connect before starting the app
        time.sleep(2)
    
    # Start simulation thread if using simulated data
    if use_simulated_data:
        sim_thread = threading.Thread(target=simulation_thread)
        sim_thread.daemon = True
        sim_thread.start()
    
    # Start Flask app
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Start the traffic control dashboard')
    parser.add_argument('--port', '-p', type=int, default=8080, help='Port to run the dashboard on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--no-sim', action='store_true', help='Disable simulated data generation')
    parser.add_argument('--debug', action='store_true', help='Run Flask in debug mode')
    
    args = parser.parse_args()
    
    print(f"Starting Traffic Control Dashboard on {args.host}:{args.port}")
    print(f"Simulated data: {'disabled' if args.no_sim else 'enabled'}")
    print(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
    
    start_dashboard(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_simulated_data=not args.no_sim
    )