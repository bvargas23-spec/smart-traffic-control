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
TIMING_TOPIC = "traffic/intersection/timing"
CONGESTION_TOPIC = "traffic/#"

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
        connect_result = mqtt_client.connect()
        if not connect_result:
            logger.error("Failed to connect to AWS IoT")
            return False
            
        logger.info("Connected to AWS IoT")
        return True

    def status_callback(client, userdata, message):
        """Callback for status messages"""
        try:
            payload = json.loads(message.payload)
            process_status_update(payload)
        except Exception as e:
            logger.error(f"Error processing status message: {e}")

    def timing_callback(client, userdata, message):
        """Callback for timing messages"""
        try:
            payload = json.loads(message.payload)
            process_timing_update(payload)
        except Exception as e:
            logger.error(f"Error processing timing message: {e}")

    def congestion_callback(client, userdata, message):
        """Callback for congestion alerts"""
        try:
            payload = json.loads(message.payload)
            process_congestion_alert(payload)
        except Exception as e:
            logger.error(f"Error processing congestion alert: {e}")

def process_status_update(payload):
    """Process a status update message"""
    global dashboard_state
    
    intersection_id = payload.get("intersection_id")
    if not intersection_id or intersection_id not in dashboard_state["intersections"]:
        return
        
    # Update intersection data
    intersection = dashboard_state["intersections"][intersection_id]
    intersection["status"] = payload.get("status", "unknown")
    intersection["role"] = payload.get("role", "unknown")
    intersection["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check for traffic conditions in the update
    if "traffic_conditions" in payload:
        intersection["traffic_conditions"] = payload["traffic_conditions"]
    
    # Update dashboard last update time
    dashboard_state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info(f"Updated status for {intersection_id}: {intersection['status']}")

def process_timing_update(payload):
    """Process a timing update message"""
    global dashboard_state
    
    intersection_id = payload.get("intersection_id")
    if not intersection_id or intersection_id not in dashboard_state["intersections"]:
        return
        
    # Update timing plan
    if "timing_plan" in payload:
        dashboard_state["intersections"][intersection_id]["timing_plan"] = payload["timing_plan"]
        
    # Add to alerts
    alert = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "timing",
        "intersection": dashboard_state["intersections"][intersection_id]["name"],
        "message": f"Timing updated: NS={payload['timing_plan'].get('NORTH_SOUTH', 'unknown')}s, EW={payload['timing_plan'].get('EAST_WEST', 'unknown')}s"
    }
    dashboard_state["alerts"].insert(0, alert)
    
    # Trim alerts list if needed
    if len(dashboard_state["alerts"]) > MAX_ALERTS:
        dashboard_state["alerts"] = dashboard_state["alerts"][:MAX_ALERTS]
    
    # Update dashboard last update time
    dashboard_state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info(f"Updated timing for {intersection_id}: {payload['timing_plan']}")

def process_congestion_alert(payload):
    """Process a congestion alert message"""
    global dashboard_state
    
    intersection_id = payload.get("intersection_id")
    direction = payload.get("direction")
    congestion_level = payload.get("congestion_level")
    
    if not all([intersection_id, direction, congestion_level]):
        return
        
    if intersection_id not in dashboard_state["intersections"]:
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

def start_dashboard(host='0.0.0.0', port=8080, debug=False, use_simulated_data=True):
    # Start MQTT client in background thread if available
    if aws_iot_available:
        mqtt_thread = threading.Thread(target=setup_mqtt_client)
        mqtt_thread.daemon = True
        mqtt_thread.start()
    
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