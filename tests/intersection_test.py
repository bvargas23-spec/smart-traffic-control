import Adafruit_BBIO.GPIO as GPIO
import time

# Set up the GPIO pins
# Using the pins from the PRU-ICSS table
RED_PIN = "P8_11"    # GPIO1_13
YELLOW_PIN = "P8_12" # GPIO1_12 
GREEN_PIN = "P8_15"  # GPIO1_15

# Set up the pins as outputs
GPIO.setup(RED_PIN, GPIO.OUT)
GPIO.setup(YELLOW_PIN, GPIO.OUT)
GPIO.setup(GREEN_PIN, GPIO.OUT)

def all_off():
    """Turn all lights off"""
    GPIO.output(RED_PIN, GPIO.LOW)
    GPIO.output(YELLOW_PIN, GPIO.LOW)
    GPIO.output(GREEN_PIN, GPIO.LOW)

try:
    # Run the traffic light sequence
    print("Starting traffic light sequence. Press Ctrl+C to stop.")
    
    while True:
        # Red light
        all_off()
        print("RED ON")
        GPIO.output(RED_PIN, GPIO.HIGH)
        time.sleep(3)
        
        # Green light
        all_off()
        print("GREEN ON")
        GPIO.output(GREEN_PIN, GPIO.HIGH)
        time.sleep(3)
        
        # Yellow light
        all_off()
        print("YELLOW ON")
        GPIO.output(YELLOW_PIN, GPIO.HIGH)
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\nTraffic light sequence stopped by user")
finally:
    # Clean up GPIO to release pins
    all_off()
    GPIO.cleanup()
    print("GPIO pins cleaned up")