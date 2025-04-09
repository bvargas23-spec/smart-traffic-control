import Adafruit_BBIO.GPIO as GPIO
import time

# First traffic light (North) - pins you're already using
NORTH_RED_PIN = "P8_11"
NORTH_YELLOW_PIN = "P8_12"
NORTH_GREEN_PIN = "P8_15"

# Second traffic light (East) - new pins
EAST_RED_PIN = "P8_7"
EAST_YELLOW_PIN = "P8_8"
EAST_GREEN_PIN = "P8_9"

# Set up all pins as outputs
GPIO.setup(NORTH_RED_PIN, GPIO.OUT)
GPIO.setup(NORTH_YELLOW_PIN, GPIO.OUT)
GPIO.setup(NORTH_GREEN_PIN, GPIO.OUT)
GPIO.setup(EAST_RED_PIN, GPIO.OUT)
GPIO.setup(EAST_YELLOW_PIN, GPIO.OUT)
GPIO.setup(EAST_GREEN_PIN, GPIO.OUT)

def all_off():
    """Turn all lights off"""
    GPIO.output(NORTH_RED_PIN, GPIO.LOW)
    GPIO.output(NORTH_YELLOW_PIN, GPIO.LOW)
    GPIO.output(NORTH_GREEN_PIN, GPIO.LOW)
    GPIO.output(EAST_RED_PIN, GPIO.LOW)
    GPIO.output(EAST_YELLOW_PIN, GPIO.LOW)
    GPIO.output(EAST_GREEN_PIN, GPIO.LOW)

try:
    # Run the intersection traffic light sequence
    print("Starting intersection control. Press Ctrl+C to stop.")
    
    while True:
        # North green, East red
        all_off()
        print("North: GREEN, East: RED")
        GPIO.output(NORTH_GREEN_PIN, GPIO.HIGH)
        GPIO.output(EAST_RED_PIN, GPIO.HIGH)
        time.sleep(5)
        
        # North yellow, East red
        all_off()
        print("North: YELLOW, East: RED")
        GPIO.output(NORTH_YELLOW_PIN, GPIO.HIGH)
        GPIO.output(EAST_RED_PIN, GPIO.HIGH)
        time.sleep(2)
        
        # North red, East green
        all_off()
        print("North: RED, East: GREEN")
        GPIO.output(NORTH_RED_PIN, GPIO.HIGH)
        GPIO.output(EAST_GREEN_PIN, GPIO.HIGH)
        time.sleep(5)
        
        # North red, East yellow
        all_off()
        print("North: RED, East: YELLOW")
        GPIO.output(NORTH_RED_PIN, GPIO.HIGH)
        GPIO.output(EAST_YELLOW_PIN, GPIO.HIGH)
        time.sleep(2)
        
except KeyboardInterrupt:
    print("\nTraffic light sequence stopped by user")
finally:
    # Clean up GPIO to release pins
    all_off()
    GPIO.cleanup()
    print("GPIO pins cleaned up")