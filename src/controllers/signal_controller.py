import Adafruit_BBIO.GPIO as GPIO
import time
import logging
import threading
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("traffic_controller.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TrafficControl")

class SignalState(Enum):
    """Enum to represent traffic light states"""
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"
    FLASHING_RED = "FLASHING_RED"
    FLASHING_YELLOW = "FLASHING_YELLOW"
    OFF = "OFF"

class Direction(Enum):
    """Enum to represent directions"""
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"

class Phase(Enum):
    """Enum to represent signal phases"""
    NORTH_SOUTH = "NORTH_SOUTH"
    EAST_WEST = "EAST_WEST"

class SignalController:
    """
    Controller for a 4-way intersection with traffic lights
    """
    
    def __init__(self, traffic_light_pins, min_green_time=10, yellow_time=3, all_red_time=2):
        """
        Initialize the signal controller
        
        Args:
            traffic_light_pins: Dict mapping directions to pin configurations
                Format: {
                    "NORTH": {"red": "P8_7", "yellow": "P8_8", "green": "P8_9"},
                    "SOUTH": {"red": "P8_10", "yellow": "P8_11", "green": "P8_12"},
                    "EAST": {"red": "P8_13", "yellow": "P8_14", "green": "P8_15"},
                    "WEST": {"red": "P8_16", "yellow": "P8_17", "green": "P8_18"}
                }
            min_green_time: Minimum green light time in seconds
            yellow_time: Yellow light time in seconds
            all_red_time: All-red safety time in seconds
        """
        self.traffic_light_pins = traffic_light_pins
        self.min_green_time = min_green_time
        self.yellow_time = yellow_time
        self.all_red_time = all_red_time
        
        # Current state tracking
        self.current_phase = None
        self.phase_start_time = 0
        self.running = False
        self.control_thread = None
        
        # Default timing plan (will be updated with adaptive timing)
        self.timing_plan = {
            Phase.NORTH_SOUTH: 30,  # seconds of green time
            Phase.EAST_WEST: 30     # seconds of green time
        }
        
        # Initialize GPIO
        self._setup_gpio()
        
        logger.info("Signal controller initialized")
    
    def _setup_gpio(self):
        """Set up GPIO pins"""
        for direction, pins in self.traffic_light_pins.items():
            GPIO.setup(pins["red"], GPIO.OUT)
            GPIO.setup(pins["yellow"], GPIO.OUT)
            GPIO.setup(pins["green"], GPIO.OUT)
            
            # Default to red
            GPIO.output(pins["red"], GPIO.HIGH)
            GPIO.output(pins["yellow"], GPIO.LOW)
            GPIO.output(pins["green"], GPIO.LOW)
        
        logger.info("GPIO pins initialized")
    
    def set_light_state(self, direction, state):
        """
        Set a specific traffic light to a given state
        
        Args:
            direction: Direction of the traffic light
            state: SignalState to set
        """
        pins = self.traffic_light_pins[direction]
        
        # Turn all lights off first
        GPIO.output(pins["red"], GPIO.LOW)
        GPIO.output(pins["yellow"], GPIO.LOW)
        GPIO.output(pins["green"], GPIO.LOW)
        
        # Set the appropriate light
        if state == SignalState.RED:
            GPIO.output(pins["red"], GPIO.HIGH)
        elif state == SignalState.YELLOW:
            GPIO.output(pins["yellow"], GPIO.HIGH)
        elif state == SignalState.GREEN:
            GPIO.output(pins["green"], GPIO.HIGH)
        elif state == SignalState.FLASHING_RED:
            GPIO.output(pins["red"], GPIO.HIGH)
            # Flashing would be handled by a separate thread
        elif state == SignalState.FLASHING_YELLOW:
            GPIO.output(pins["yellow"], GPIO.HIGH)
            # Flashing would be handled by a separate thread
        # For OFF state, all lights remain off
        
        logger.debug(f"Set {direction} to {state.value}")
    
    def set_phase(self, phase, green_time=None):
        """
        Set the intersection to a specific phase
        
        Args:
            phase: Phase to set
            green_time: Override green time for this phase (if None, use timing plan)
        """
        # Use default timing if not specified
        if green_time is None:
            green_time = self.timing_plan[phase]
        
        # Enforce minimum green time
        green_time = max(green_time, self.min_green_time)
        
        # Set lights based on phase
        if phase == Phase.NORTH_SOUTH:
            # North-South green, East-West red
            self.set_light_state(Direction.NORTH.value, SignalState.GREEN)
            self.set_light_state(Direction.SOUTH.value, SignalState.GREEN)
            self.set_light_state(Direction.EAST.value, SignalState.RED)
            self.set_light_state(Direction.WEST.value, SignalState.RED)
        elif phase == Phase.EAST_WEST:
            # East-West green, North-South red
            self.set_light_state(Direction.NORTH.value, SignalState.RED)
            self.set_light_state(Direction.SOUTH.value, SignalState.RED)
            self.set_light_state(Direction.EAST.value, SignalState.GREEN)
            self.set_light_state(Direction.WEST.value, SignalState.GREEN)
        
        self.current_phase = phase
        self.phase_start_time = time.time()
        logger.info(f"Set phase to {phase.value} with green time {green_time}s")
        
        return green_time
    
    def transition_to_phase(self, next_phase, green_time=None):
        """
        Safely transition from current phase to next phase
        
        Args:
            next_phase: Phase to transition to
            green_time: Green time for the next phase
        """
        if self.current_phase is None:
            # First time initialization
            return self.set_phase(next_phase, green_time)
        
        # Don't transition if already in the target phase
        if self.current_phase == next_phase:
            return
        
        # Yellow transition for current green directions
        if self.current_phase == Phase.NORTH_SOUTH:
            self.set_light_state(Direction.NORTH.value, SignalState.YELLOW)
            self.set_light_state(Direction.SOUTH.value, SignalState.YELLOW)
        else:  # EAST_WEST
            self.set_light_state(Direction.EAST.value, SignalState.YELLOW)
            self.set_light_state(Direction.WEST.value, SignalState.YELLOW)
        
        # Wait for yellow time
        logger.info(f"Yellow transition for {self.yellow_time}s")
        time.sleep(self.yellow_time)
        
        # All-red safety period
        self.set_light_state(Direction.NORTH.value, SignalState.RED)
        self.set_light_state(Direction.SOUTH.value, SignalState.RED)
        self.set_light_state(Direction.EAST.value, SignalState.RED)
        self.set_light_state(Direction.WEST.value, SignalState.RED)
        
        logger.info(f"All-red safety period for {self.all_red_time}s")
        time.sleep(self.all_red_time)
        
        # Set the next phase
        return self.set_phase(next_phase, green_time)
    
    def update_timing_plan(self, new_timing_plan):
        """
        Update the signal timing plan
        
        Args:
            new_timing_plan: Dict mapping phases to green times
        """
        # Validate the new timing plan
        for phase, time in new_timing_plan.items():
            if time < self.min_green_time:
                logger.warning(f"Green time for {phase.value} less than minimum. Using {self.min_green_time}s instead of {time}s")
                new_timing_plan[phase] = self.min_green_time
        
        self.timing_plan = new_timing_plan
        logger.info(f"Updated timing plan: {self.timing_plan}")
    
    def run_cycle(self):
        """Run a complete signal cycle once"""
        # Start with North-South phase
        ns_green_time = self.transition_to_phase(Phase.NORTH_SOUTH)
        time.sleep(ns_green_time)
        
        # Then East-West phase
        ew_green_time = self.transition_to_phase(Phase.EAST_WEST)
        time.sleep(ew_green_time)
    
    def start_control_loop(self):
        """Start the signal control loop in a background thread"""
        if self.running:
            logger.warning("Control loop already running")
            return False
        
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        logger.info("Started control loop")
        return True
    
    def _control_loop(self):
        """Main control loop - runs continuously until stopped"""
        try:
            while self.running:
                self.run_cycle()
        except Exception as e:
            logger.error(f"Error in control loop: {e}")
            self.running = False
            # Set all lights to flashing red in case of error
            self.set_emergency_mode()
    
    def stop_control_loop(self):
        """Stop the signal control loop"""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=5.0)
            self.control_thread = None
        logger.info("Stopped control loop")
    
    def set_emergency_mode(self):
        """Set all signals to flashing red (emergency mode)"""
        # Stop normal operation
        self.stop_control_loop()
        
        # Set all directions to red
        for direction in Direction:
            self.set_light_state(direction.value, SignalState.RED)
        
        logger.warning("Emergency mode activated - all signals set to red")
    
    def cleanup(self):
        """Clean up resources and reset GPIO"""
        self.stop_control_loop()
        
        # Turn off all lights
        for direction, pins in self.traffic_light_pins.items():
            GPIO.output(pins["red"], GPIO.LOW)
            GPIO.output(pins["yellow"], GPIO.LOW)
            GPIO.output(pins["green"], GPIO.LOW)
        
        GPIO.cleanup()
        logger.info("Signal controller cleaned up")