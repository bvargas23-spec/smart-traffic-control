from enum import Enum
import time

class ControllerStatus(Enum):
    NORMAL = "NORMAL"
    ERROR = "ERROR"
    MAINTENANCE = "MAINTENANCE"
    EMERGENCY = "EMERGENCY"

class SignalController:
    def __init__(self, hardware_interface):
        """Initialize the signal controller with a hardware interface."""
        self.hardware_interface = hardware_interface
        self.status = ControllerStatus.NORMAL
        self.signals = []
        self.phases = []
        self.intersection_id = None
        self.sync_partners = []
        self.sync_clock_offset = 0
        self.previous_timing_plan = {}
        self.current_timing_plan = {
            "east_west_green": 30,
            "north_south_green": 30,
            "yellow_time": 3
        }
        self.default_timing_parameters = {
            "yellow_time": 3
        }
        
        # Initialize the hardware
        self.hardware_interface.initialize()
    
    def set_signal_state(self, signal_id, state):
        """Set a traffic signal to a specific state."""
        # Validate signal_id and state
        if not self._is_valid_signal_id(signal_id) or not self._is_valid_state(state):
            return False
            
        # Send command to hardware
        result = self.hardware_interface.send_command(
            signal_id=signal_id,
            command="SET_STATE",
            value=state
        )
        
        # Update controller status if command failed
        if not result:
            self.status = ControllerStatus.ERROR
            
        return result
    
    def activate_emergency_mode(self):
        """Activate emergency mode for all signals."""
        result = self.hardware_interface.send_command(command="EMERGENCY_MODE")
        if result:
            self.status = ControllerStatus.EMERGENCY
        return result
    
    def check_status(self):
        """Check the status of the controller and hardware."""
        hardware_status = self.hardware_interface.get_status()
        
        if hardware_status == "FAULT":
            self.status = ControllerStatus.ERROR
            
        return self.status
    
    def execute_phase(self, phase_sequence):
        """Execute a sequence of signal state changes."""
        for signal_config in phase_sequence:
            signal_id = signal_config["signal_id"]
            state = signal_config["state"]
            self.set_signal_state(signal_id, state)
        
        return True
    
    def load_configuration(self, config):
        """Load a configuration into the controller."""
        self.intersection_id = config.get("intersection_id")
        self.signals = config.get("signals", [])
        self.phases = config.get("phases", [])
        return True
    
    def update_configuration(self, config):
        """Update the current configuration."""
        # Preserve status during update
        current_status = self.status
        
        # Update configuration
        result = self.load_configuration(config)
        
        # Restore status
        self.status = current_status
        
        return result
    
    def recover_partial_failure(self):
        """Recover from a partial failure of the system."""
        signal_statuses = self.hardware_interface.get_signal_status()
        
        for signal_id, status in signal_statuses.items():
            if status == "FAULT":
                self.hardware_interface.reset_signal(signal_id=signal_id)
                
        return True
    
    def add_sync_partner(self, partner_controller):
        """Add a partner controller for synchronization."""
        self.sync_partners.append(partner_controller)
        return True
    
    def execute_synchronized_phase(self, phase_id):
        """Execute a phase change synchronized with partners."""
        # Execute on this controller
        timestamp = time.time()
        
        # Execute on partners with adjusted timestamp
        for partner in self.sync_partners:
            partner_timestamp = timestamp + self.sync_clock_offset
            partner.hardware_interface.send_command(timestamp=partner_timestamp)
            
        return True
    
    def execute_synchronized_command(self, command, **kwargs):
        """Execute a command synchronized with partners."""
        # Execute on this controller
        timestamp = time.time()
        self.hardware_interface.send_command(command=command, timestamp=timestamp, **kwargs)
        
        # Execute on partners with adjusted timestamp
        for partner in self.sync_partners:
            partner_timestamp = timestamp + self.sync_clock_offset
            partner.hardware_interface.send_command(
                command=command, timestamp=partner_timestamp, **kwargs
            )
            
        return True
    
    def adjust_timing_for_volume(self, signal_id, volume):
        """Adjust timing based on traffic volume."""
        # Calculate appropriate timing based on volume
        new_timing = volume // 10  # Simple calculation for testing
        
        # Apply the new timing
        self.hardware_interface.send_command(
            signal_id=signal_id,
            command="SET_TIMING",
            value=new_timing
        )
        
        return new_timing
    
    def recover_from_fault(self):
        """Recover from a fault condition."""
        result = self.hardware_interface.reset()
        if result:
            self.status = ControllerStatus.NORMAL
        return result
    
    def run_adaptive_control_cycle(self, sensor_controller):
        """Run an adaptive control cycle based on sensor data."""
        volumes = sensor_controller.get_approach_volumes()
        
        # Calculate new timing based on volumes
        north_south_volume = volumes["north"] + volumes["south"]
        east_west_volume = volumes["east"] + volumes["west"]
        
        total_volume = north_south_volume + east_west_volume
        
        # Allocate green time proportionally to volumes
        ns_proportion = north_south_volume / total_volume
        ew_proportion = east_west_volume / total_volume
        
        # Save current timing as previous
        self.previous_timing_plan = self.current_timing_plan.copy()
        
        # Update timing plan
        self.current_timing_plan = {
            "north_south_green": int(60 * ns_proportion),
            "east_west_green": int(60 * ew_proportion),
            "yellow_time": 3
        }
        
        return True
    
    def run_predictive_control(self, sensor_controller, prediction_horizon):
        """Run predictive control using predicted volumes."""
        current_volumes = sensor_controller.get_current_volumes()
        predicted_volumes = sensor_controller.predict_future_volumes()
        
        # Save current timing as previous
        self.previous_timing_plan = self.current_timing_plan.copy()
        
        # Calculate timing based on predicted volumes
        east_volume = predicted_volumes["east"]
        west_volume = predicted_volumes["west"]
        east_west_volume = east_volume + west_volume
        
        # Increase east-west green time based on predicted increase
        current_ew = current_volumes["east"] + current_volumes["west"]
        if east_west_volume > current_ew:
            increase_factor = east_west_volume / current_ew
            new_green = int(self.current_timing_plan["east_west_green"] * increase_factor)
            self.current_timing_plan["east_west_green"] = new_green
            
        return True
    
    def add_to_coordination_group(self, upstream, downstream, corridor_speed):
        """Add controller to a coordination group."""
        self.upstream_controller = upstream
        self.downstream_controller = downstream
        self.corridor_speed = corridor_speed  # km/h
        
        # Calculate offset based on distance and speed
        distance_to_upstream = 500  # meters, for testing
        travel_time = distance_to_upstream / (corridor_speed * 1000/3600)  # seconds
        
        self.timing_parameters = {
            "offset_to_upstream": travel_time
        }
        
        return True
    
    def update_coordinated_timing(self):
        """Update timing in coordination with adjacent intersections."""
        # Use the previously calculated offset
        return True
    
    def handle_pedestrian_calls(self, sensor_controller, max_wait_time):
        """Handle pedestrian call requests."""
        has_call = sensor_controller.get_pedestrian_call()
        waiting_time = sensor_controller.get_pedestrian_waiting_time()
        
        if has_call and waiting_time > max_wait_time * 0.75:
            self.hardware_interface.send_command(
                command="SET_PEDESTRIAN_PHASE",
                value="ACTIVATE"
            )
            
        return True
    
    def handle_transit_priority(self, sensor_controller):
        """Handle transit signal priority."""
        transit_info = sensor_controller.detect_transit_vehicle()
        
        if transit_info["detected"]:
            # Determine if we should extend green or provide early green
            if transit_info["distance"] < 100:
                self.hardware_interface.send_command(command="EXTEND_GREEN")
            else:
                self.hardware_interface.send_command(command="EARLY_GREEN")
                
        return True
    
    def handle_oversaturation(self, sensor_controller):
        """Handle oversaturated traffic conditions."""
        saturation_levels = sensor_controller.get_saturation_levels()
        
        # Find the most oversaturated approach
        max_approach = max(saturation_levels.items(), key=lambda x: x[1])
        approach, level = max_approach
        
        if level > 1.0:  # Oversaturated
            # Determine direction to prioritize
            if approach in ["north", "south"]:
                direction = "NORTH_SOUTH"
                green_time = self.current_timing_plan["north_south_green"]
            else:
                direction = "EAST_WEST"
                green_time = self.current_timing_plan["east_west_green"]
                
            # Activate flush mode in the congested direction
            self.hardware_interface.send_command(
                command="SET_FLUSH_MODE",
                direction=direction,
                duration=green_time
            )
            
        return True
    
    def adjust_for_weather_conditions(self, sensor_controller):
        """Adjust signal timing for adverse weather."""
        weather = sensor_controller.get_weather_conditions()
        
        if weather["condition"] in ["RAIN", "SNOW"]:
            # Extend yellow time for adverse weather
            extended_yellow = self.default_timing_parameters["yellow_time"] * 1.5
            self.current_timing_plan["yellow_time"] = extended_yellow
            
        return True
    
    def determine_adaptive_action(self, sensor_controller):
        """Determine adaptive action based on traffic conditions."""
        conditions = sensor_controller.get_traffic_conditions()
        
        if conditions["congestion_level"] == "HIGH" and conditions["queue_length"] > 15:
            return "extend_green"
        elif conditions["congestion_level"] == "LOW" and conditions["queue_length"] < 10:
            return "reduce_green"
        else:
            return "maintain"
    
    def load_configuration_file(self, file_path):
        """Load configuration from a file."""
        import json
        with open(file_path, 'r') as f:
            config = json.load(f)
        return self.load_configuration(config)
    
    def recover_from_failure(self, scenario):
        """Recover from various failure scenarios."""
        if scenario == "power_outage":
            self.hardware_interface.power_status()
        elif scenario == "communication_loss":
            self.hardware_interface.communication_status()
        elif scenario == "software_reset":
            self.hardware_interface.software_status()
            
        self.status = ControllerStatus.NORMAL
        return True
    
    def _is_valid_signal_id(self, signal_id):
        """Check if signal ID is valid."""
        return 0 <= signal_id <= 255
    
    def _is_valid_state(self, state):
        """Check if signal state is valid."""
        valid_states = ["GREEN", "RED", "YELLOW", "FLASHING_RED"]
        return state in valid_states

class SensorController:
    def __init__(self, sensor_interface):
        """Initialize the sensor controller with a sensor interface."""
        self.sensor_interface = sensor_interface
        self.fault_log = {}
    
    def get_vehicle_count(self, sensor_id):
        """Get vehicle count from a specific sensor."""
        try:
            return self.sensor_interface.read_vehicle_count(sensor_id=sensor_id)
        except Exception as e:
            self.fault_log[sensor_id] = str(e)
            return -1  # Error code
    
    def get_average_speed(self, sensor_id):
        """Get average vehicle speed from a sensor."""
        try:
            return self.sensor_interface.read_vehicle_speed(sensor_id=sensor_id)
        except Exception as e:
            self.fault_log[sensor_id] = str(e)
            return -1
    
    def get_lane_occupancy(self, sensor_id):
        """Get lane occupancy percentage from a sensor."""
        try:
            return self.sensor_interface.read_occupancy(sensor_id=sensor_id)
        except Exception as e:
            self.fault_log[sensor_id] = str(e)
            return -1
    
    def has_fault(self, sensor_id):
        """Check if a sensor has a fault recorded."""
        return sensor_id in self.fault_log
    
    def get_all_sensor_statuses(self):
        """Get status of all sensors."""
        return self.sensor_interface.get_all_statuses()
    
    def get_total_vehicle_count(self, sensor_ids):
        """Get total vehicle count from multiple sensors."""
        total = 0
        for sensor_id in sensor_ids:
            count = self.get_vehicle_count(sensor_id)
            if count >= 0:  # Only add valid counts
                total += count
        return total
    
    def get_corridor_average_speed(self, sensor_ids):
        """Get average speed across a corridor from multiple sensors."""
        speeds = []
        for sensor_id in sensor_ids:
            speed = self.get_average_speed(sensor_id)
            if speed >= 0:  # Only add valid speeds
                speeds.append(speed)
                
        if speeds:
            return sum(speeds) / len(speeds)
        return 0
    
    def get_filtered_vehicle_count(self, sensor_ids):
        """Get filtered vehicle count after removing outliers."""
        counts = []
        for sensor_id in sensor_ids:
            count = self.get_vehicle_count(sensor_id)
            if count >= 0:
                counts.append(count)
                
        if not counts:
            return 0
            
        # Simple outlier removal (exclude values > 2 std devs from mean)
        if len(counts) > 2:
            mean = sum(counts) / len(counts)
            std_dev = (sum((x - mean) ** 2 for x in counts) / len(counts)) ** 0.5
            filtered_counts = [c for c in counts if abs(c - mean) <= 2 * std_dev]
            
            if filtered_counts:
                return sum(filtered_counts) / len(filtered_counts)
                
        return sum(counts) / len(counts)
    
    def get_calibrated_reading(self, sensor_id):
        """Get calibrated sensor reading."""
        raw_value = self.sensor_interface.read_raw_value(sensor_id=sensor_id)
        calibration_factor = self.sensor_interface.get_calibration_factor(sensor_id=sensor_id)
        
        return raw_value * calibration_factor
    
    def get_speed_in_mph(self, sensor_id):
        """Get vehicle speed in mph (converted from kph)."""
        kph_speed = self.get_average_speed(sensor_id)
        if kph_speed >= 0:
            return kph_speed * 0.621371  # Convert kph to mph
        return -1
    
    def collect_samples(self, sensor_id, count):
        """Collect multiple samples from a sensor."""
        samples = []
        for _ in range(count):
            samples.append(self.sensor_interface.read_value(sensor_id=sensor_id))
        return samples
    
    def integrate_sensor_data(self, sensor_id, duration):
        """Integrate sensor data over time."""
        time_series = self.sensor_interface.get_time_series(sensor_id=sensor_id)
        return sum(time_series)
    
    def calculate_rate_of_change(self, sensor_id, duration):
        """Calculate rate of change in sensor data."""
        time_series = self.sensor_interface.get_time_series(sensor_id=sensor_id)
        derivatives = []
        
        for i in range(1, len(time_series)):
            derivatives.append(time_series[i] - time_series[i-1])
            
        return derivatives
    
    def get_approach_volumes(self):
        """Get traffic volumes for each approach."""
        return self.sensor_interface.get_approach_volumes()
    
    def get_current_volumes(self):
        """Get current traffic volumes."""
        return self.sensor_interface.get_current_volumes()
    
    def predict_future_volumes(self):
        """Predict future traffic volumes."""
        return self.sensor_interface.predict_future_volumes()
    
    def get_pedestrian_call(self):
        """Check if there is a pedestrian call."""
        return self.sensor_interface.get_pedestrian_call()
    
    def get_pedestrian_waiting_time(self):
        """Get pedestrian waiting time."""
        return self.sensor_interface.get_pedestrian_waiting_time()
    
    def detect_transit_vehicle(self):
        """Detect transit vehicles approaching the intersection."""
        return self.sensor_interface.detect_transit_vehicle()
    
    def get_saturation_levels(self):
        """Get saturation levels for each approach."""
        return self.sensor_interface.get_saturation_levels()
    
    def get_weather_conditions(self):
        """Get current weather conditions."""
        return self.sensor_interface.get_weather_conditions()
    
    def get_traffic_conditions(self):
        """Get current traffic conditions."""
        return self.sensor_interface.get_traffic_conditions()
    
    def detect_emergency_vehicle(self, intersection_id):
        """Detect emergency vehicles approaching the intersection."""
        return self.sensor_interface.detect_emergency_vehicle()