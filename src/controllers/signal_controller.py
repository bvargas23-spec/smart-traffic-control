#!/usr/bin/env python3
# signal_controller.py - Add missing get_timing_plan method for intersection_coordinator.py

import Adafruit_BBIO.GPIO as GPIO
import time
import threading
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TrafficControl")

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

class SignalController:
    """Controller for traffic signal hardware"""
    
    def __init__(self, traffic_light_pins, min_green_time=15, yellow_time=3, all_red_time=2):
        """
        Initialize the signal controller
        
        Args:
            traffic_light_pins: Dictionary mapping directions to GPIO pins
            min_green_time: Minimum green time in seconds
            yellow_time: Yellow time in seconds
            all_red_time: All-red safety interval in seconds
        """
        self.traffic_light_pins = traffic_light_pins
        self.min_green_time = min_green_time
        self.yellow_time = yellow_time
        self.all_red_time = all_red_time
        
        # Set up timing plan
        self.timing_plan = {
            Phase.NORTH_SOUTH: 30,  # Default values
            Phase.EAST_WEST: 30
        }
        
        # Set up GPIO
        for direction, pins in self.traffic_light_pins.items():
            GPIO.setup(pins["red"], GPIO.OUT)
            GPIO.setup(pins["yellow"], GPIO.OUT)
            GPIO.setup(pins["green"], GPIO.OUT)
            
            # Initialize with all lights off
            GPIO.output(pins["red"], GPIO.LOW)
            GPIO.output(pins["yellow"], GPIO.LOW)
            GPIO.output(pins["green"], GPIO.LOW)
        
        # Set initial phase
        self.current_phase = None
        self.running = False
        self.control_thread = None
        
        logger.info("GPIO pins initialized")
        logger.info("Signal controller initialized")
    
    def all_lights_off(self):
        """Turn off all traffic lights"""
        for direction, pins in self.traffic_light_pins.items():
            GPIO.output(pins["red"], GPIO.LOW)
            GPIO.output(pins["yellow"], GPIO.LOW)
            GPIO.output(pins["green"], GPIO.LOW)
    
    def set_signal(self, direction, state):
        """
        Set a specific signal to a given state
        
        Args:
            direction: Direction enum or string
            state: SignalState enum or string
        """
        # Convert direction to string if it's an enum
        if isinstance(direction, Direction):
            direction = direction.value
            
        # Convert state to string if it's an enum
        if isinstance(state, SignalState):
            state = state.value
        
        # Get pins for this direction
        pins = self.traffic_light_pins.get(direction)
        if not pins:
            logger.error(f"Unknown direction: {direction}")
            return
        
        # Turn off all lights for this direction
        GPIO.output(pins["red"], GPIO.LOW)
        GPIO.output(pins["yellow"], GPIO.LOW)
        GPIO.output(pins["green"], GPIO.LOW)
        
        # Turn on the requested state
        if state == "RED" or state == SignalState.RED:
            GPIO.output(pins["red"], GPIO.HIGH)
        elif state == "YELLOW" or state == SignalState.YELLOW:
            GPIO.output(pins["yellow"], GPIO.HIGH)
        elif state == "GREEN" or state == SignalState.GREEN:
            GPIO.output(pins["green"], GPIO.HIGH)
        else:
            logger.error(f"Unknown state: {state}")
    
    def set_phase(self, phase, green_time=None):
        """
        Set the current phase (which directions get green)
        
        Args:
            phase: Phase enum
            green_time: Green time in seconds, or None to use timing_plan
        """
        if green_time is None:
            # Use timing plan if no specific time provided
            green_time = self.timing_plan[phase]
        
        # Set current phase
        self.current_phase = phase
        
        # Set signals based on phase
        if phase == Phase.NORTH_SOUTH:
            # North-South gets green, East-West gets red
            self.set_signal(Direction.NORTH, SignalState.GREEN)
            self.set_signal(Direction.SOUTH, SignalState.GREEN)
            self.set_signal(Direction.EAST, SignalState.RED)
            self.set_signal(Direction.WEST, SignalState.RED)
            logger.info(f"Set phase to NORTH_SOUTH with green time {green_time}s")
        elif phase == Phase.EAST_WEST:
            # East-West gets green, North-South gets red
            self.set_signal(Direction.NORTH, SignalState.RED)
            self.set_signal(Direction.SOUTH, SignalState.RED)
            self.set_signal(Direction.EAST, SignalState.GREEN)
            self.set_signal(Direction.WEST, SignalState.GREEN)
            logger.info(f"Set phase to EAST_WEST with green time {green_time}s")
        else:
            logger.error(f"Unknown phase: {phase}")
            return 0
            
        return green_time
    
    def transition_to_phase(self, next_phase):
        """
        Transition from current phase to next phase with proper yellow and all-red
        
        Args:
            next_phase: The phase to transition to
        
        Returns:
            int: Green time for the new phase
        """
        # Figure out which directions are transitioning from green to red
        transitioning_directions = []
        if self.current_phase == Phase.NORTH_SOUTH and next_phase == Phase.EAST_WEST:
            transitioning_directions = [Direction.NORTH, Direction.SOUTH]
        elif self.current_phase == Phase.EAST_WEST and next_phase == Phase.NORTH_SOUTH:
            transitioning_directions = [Direction.EAST, Direction.WEST]
        
        # Yellow transition
        for direction in transitioning_directions:
            self.set_signal(direction, SignalState.YELLOW)
        logger.info(f"Yellow transition for {self.yellow_time}s")
        time.sleep(self.yellow_time)
        
        # All-red safety period
        for direction in transitioning_directions:
            self.set_signal(direction, SignalState.RED)
        logger.info(f"All-red safety period for {self.all_red_time}s")
        time.sleep(self.all_red_time)
        
        # Now set the new phase
        return self.set_phase(next_phase)
    
    def run_cycle(self):
        """Run a complete signal cycle"""
        try:
            # Run appropriate cycle based on current phase
            if self.current_phase == Phase.NORTH_SOUTH:
                # Hold green for the specified time
                time.sleep(self.timing_plan[Phase.NORTH_SOUTH])
                # Transition to East-West
                ew_green_time = self.transition_to_phase(Phase.EAST_WEST)
                # Hold East-West for its time
                time.sleep(ew_green_time)
                # Transition back to North-South
                self.transition_to_phase(Phase.NORTH_SOUTH)
            else:  # EAST_WEST or None
                # Set phase to NORTH_SOUTH if not set
                if self.current_phase is None:
                    self.set_phase(Phase.NORTH_SOUTH)
                else:
                    # Hold green for the specified time if already in EAST_WEST
                    time.sleep(self.timing_plan[Phase.EAST_WEST])
                    # Transition to North-South
                    ns_green_time = self.transition_to_phase(Phase.NORTH_SOUTH)
                    # Hold North-South for its time
                    time.sleep(ns_green_time)
                    # Transition back to East-West
                    self.transition_to_phase(Phase.EAST_WEST)
        except Exception as e:
            logger.error(f"Error in run_cycle: {e}")
            raise
    
    def _control_loop(self):
        """Main control loop for traffic signal operation"""
        # If phase not set, default to NORTH_SOUTH
        if self.current_phase is None:
            self.set_phase(Phase.NORTH_SOUTH)
        
        # Run indefinitely until stopped
        while self.running:
            try:
                self.run_cycle()
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
                # Go to emergency mode
                self.set_emergency_mode()
                break
    
    def stop_control_loop(self):
        """Stop the control loop thread"""
        self.running = False
        if self.control_thread and self.control_thread != threading.current_thread():
            self.control_thread.join(timeout=5.0)
    
    def set_emergency_mode(self):
        """Set all signals to flashing red (emergency mode)"""
        self.stop_control_loop()
        
        # All directions flashing red
        while True:  # Keep flashing until manually reset
            # All red on
            for direction in self.traffic_light_pins:
                self.set_signal(direction, SignalState.RED)
            time.sleep(0.5)
            
            # All off
            self.all_lights_off()
            time.sleep(0.5)
    
    def start_control_loop(self):
        """Start the signal control loop in a background thread"""
        if self.running:
            logger.warning("Control loop already running")
            return
            
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        logger.info("Started control loop")
    
    def update_timing_plan(self, timing_plan):
        """
        Update the signal timing plan
        
        Args:
            timing_plan: Dictionary mapping phases to green times
        """
        # Validate the timing plan
        for phase in Phase:
            if phase not in timing_plan:
                logger.warning(f"Missing phase {phase} in timing plan, using default")
                timing_plan[phase] = self.timing_plan.get(phase, 30)
            elif timing_plan[phase] < self.min_green_time:
                logger.warning(f"Green time for {phase} too short, using minimum {self.min_green_time}s")
                timing_plan[phase] = self.min_green_time
        
        # Update timing plan
        self.timing_plan = timing_plan
        logger.info(f"Updated timing plan: {timing_plan}")
        return True
        
    def get_timing_plan(self):
        """
        Get the current timing plan
        
        Returns:
            dict: Current timing plan
        """
        return self.timing_plan
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_control_loop()
        self.all_lights_off()
        GPIO.cleanup()
        logger.info("GPIO cleaned up")

if __name__ == "__main__":
    # Example use
    traffic_light_pins = {
        "NORTH": {"red": "P8_7", "yellow": "P8_8", "green": "P8_9"},
        "SOUTH": {"red": "P8_10", "yellow": "P8_11", "green": "P8_12"},
        "EAST": {"red": "P8_13", "yellow": "P8_14", "green": "P8_15"},
        "WEST": {"red": "P8_16", "yellow": "P8_17", "green": "P8_18"}
    }
    
    controller = SignalController(traffic_light_pins)
    
    try:
        # Start the control loop
        controller.start_control_loop()
        
        # Run for 2 minutes
        print("Running for 2 minutes...")
        time.sleep(60)
        
        # Update timing plan
        print("Updating timing plan...")
        controller.update_timing_plan({
            Phase.NORTH_SOUTH: 40,
            Phase.EAST_WEST: 20
        })
        
        # Run for another minute
        print("Running for another minute...")
        time.sleep(60)
        
    except KeyboardInterrupt:
        print("Stopping on keyboard interrupt")
    finally:
        # Clean up
        controller.cleanup()