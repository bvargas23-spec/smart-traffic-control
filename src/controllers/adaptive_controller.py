import os
import time
import json
import logging
from datetime import datetime
from signal_controller import SignalController, Phase, Direction, SignalState
from src.api.tomtom_client import TomTomClient
from src.api.data_processor import DataProcessor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("adaptive_controller.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AdaptiveControl")

class AdaptiveSignalController:
    """
    Adaptive traffic signal controller that uses TomTom traffic data
    to dynamically adjust signal timing
    """
    
    def __init__(self, intersection_lat, intersection_lon, update_interval=300):
        """
        Initialize the adaptive controller
        
        Args:
            intersection_lat (float): Latitude of the intersection
            intersection_lon (float): Longitude of the intersection
            update_interval (int): Seconds between traffic data updates
        """
        # Initialize TomTom API client
        api_key = os.getenv("TOMTOM_API_KEY")
        if not api_key:
            raise ValueError("No TomTom API key found in environment variables")
        
        self.tomtom_client = TomTomClient(api_key=api_key)
        self.data_processor = DataProcessor()
        
        # Set up intersection coordinates
        self.intersection_lat = intersection_lat
        self.intersection_lon = intersection_lon
        
        # Map directions to geographic approaches
        self.direction_mapping = {
            Direction.NORTH: "North",
            Direction.SOUTH: "South",
            Direction.EAST: "East",
            Direction.WEST: "West"
        }
        
        # Track the update interval
        self.update_interval = update_interval
        self.last_update_time = 0
        
        # Create the signal controller with our pin configuration
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
        
        # Data logging
        self.data_log = []
        self.log_file = "traffic_data_log.json"
        
        logger.info(f"Adaptive controller initialized for intersection at {intersection_lat}, {intersection_lon}")
    
    def fetch_traffic_data(self):
        """Fetch current traffic data from TomTom API"""
        try:
            # Get traffic summary
            traffic_summary = self.tomtom_client.get_traffic_summary(
                self.intersection_lat, 
                self.intersection_lon
            )
            
            # Process the data
            processed_data = self.data_processor.process_intersection_data({
                "intersection": traffic_summary["intersection"],
                "approaches": traffic_summary["approaches"]
            })
            
            # Log the successful data fetch
            logger.info(f"Fetched traffic data with {len(processed_data['approaches'])} approaches")
            
            # Append to data log
            self.data_log.append({
                "timestamp": datetime.now().isoformat(),
                "data": processed_data
            })
            
            # Periodically save the log
            if len(self.data_log) % 10 == 0:
                self.save_data_log()
                
            return processed_data
            
        except Exception as e:
            logger.error(f"Error fetching traffic data: {e}")
            return None
    
    def calculate_timing_plan(self, traffic_data):
        """
        Calculate optimal signal timing based on traffic data
        
        Args:
            traffic_data: Processed traffic data
            
        Returns:
            Dict mapping phases to green times
        """
        # If no data, use default timing
        if not traffic_data or "error" in traffic_data:
            logger.warning("Using default timing due to missing traffic data")
            return {
                Phase.NORTH_SOUTH: 30,
                Phase.EAST_WEST: 30
            }
        
        # Use the data processor to calculate optimal cycle times
        optimal_timing = self.data_processor.calculate_optimal_cycle_times(traffic_data)
        
        # Extract the timing information
        phase_times = optimal_timing.get("phase_times", {})
        
        # Create the new timing plan
        timing_plan = {
            Phase.NORTH_SOUTH: phase_times.get("North-South", {}).get("green_time", 30),
            Phase.EAST_WEST: phase_times.get("East-West", {}).get("green_time", 30)
        }
        
        # Log the new timing plan
        logger.info(f"Calculated timing plan: NS={timing_plan[Phase.NORTH_SOUTH]}s, EW={timing_plan[Phase.EAST_WEST]}s")
        
        return timing_plan
    
    def update_timing(self):
        """Update signal timing based on latest traffic data"""
        # Check if it's time to update
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            # Not time yet
            return
        
        # Fetch and process traffic data
        traffic_data = self.fetch_traffic_data()
        if not traffic_data:
            logger.warning("Failed to update timing due to missing traffic data")
            return
        
        # Calculate new timing plan
        timing_plan = self.calculate_timing_plan(traffic_data)
        
        # Update the signal controller
        self.signal_controller.update_timing_plan(timing_plan)
        
        # Update the last update time
        self.last_update_time = current_time
        logger.info("Successfully updated signal timing")
    
    def start(self):
        """Start the adaptive signal control system"""
        try:
            # Initial traffic data fetch and timing calculation
            logger.info("Performing initial traffic data fetch...")
            traffic_data = self.fetch_traffic_data()
            if traffic_data:
                timing_plan = self.calculate_timing_plan(traffic_data)
                self.signal_controller.update_timing_plan(timing_plan)
            
            # Start the signal controller
            self.signal_controller.start_control_loop()
            
            # Main control loop - periodically updates timing
            logger.info(f"Starting main control loop with {self.update_interval}s update interval")
            self.last_update_time = time.time()
            
            while True:
                # Update timing if needed
                self.update_timing()
                
                # Sleep for a short period to avoid busy waiting
                time.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("Adaptive controller stopped by user")
        except Exception as e:
            logger.error(f"Error in adaptive controller: {e}")
        finally:
            self.cleanup()
    
    def save_data_log(self):
        """Save the data log to a file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.data_log, f, indent=2)
            logger.info(f"Saved data log to {self.log_file}")
        except Exception as e:
            logger.error(f"Error saving data log: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        # Save any remaining data
        self.save_data_log()
        
        # Clean up the signal controller
        if hasattr(self, 'signal_controller'):
            self.signal_controller.cleanup()
        
        logger.info("Adaptive controller cleaned up")

# Main entry point
if __name__ == "__main__":
    # Roswell Road & Hwy 41 Intersection coordinates
    intersection_lat = 33.960192828395996
    intersection_lon = -84.52790520126695
    
    # Create and start the adaptive controller
    controller = AdaptiveSignalController(
        intersection_lat=intersection_lat,
        intersection_lon=intersection_lon,
        update_interval=300  # Update every 5 minutes
    )
    
    # Start the controller
    controller.start()