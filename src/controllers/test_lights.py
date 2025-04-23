import time
from signal_controller import SignalController, Phase, Direction, SignalState

# Define the pin configuration
traffic_light_pins = {
    "NORTH": {"red": "P8_7", "yellow": "P8_8", "green": "P8_9"},
    "SOUTH": {"red": "P8_10", "yellow": "P8_11", "green": "P8_12"},
    "EAST": {"red": "P8_13", "yellow": "P8_14", "green": "P8_15"},
    "WEST": {"red": "P8_16", "yellow": "P8_17", "green": "P8_18"}
}

try:
    # Create the controller
    controller = SignalController(traffic_light_pins)
    
    print("Running traffic light test sequence...")
    
    # Run 3 complete cycles
    for i in range(3):
        print(f"Cycle {i+1}")
        controller.run_cycle()
        
    print("Test complete!")
        
except KeyboardInterrupt:
    print("\nTest stopped by user")
finally:
    if 'controller' in locals():
        controller.cleanup()