import Adafruit_BBIO.GPIO as GPIO
import time

# Define GPIO pin mappings for each traffic light
traffic_lights = [
    {"red": "P8_7",  "yellow": "P8_8",  "green": "P8_9"},   # Traffic Light 1
    {"red": "P8_10", "yellow": "P8_11", "green": "P8_12"},  # Traffic Light 2
    {"red": "P8_13", "yellow": "P8_14", "green": "P8_15"},  # Traffic Light 3
    {"red": "P8_16", "yellow": "P8_17", "green": "P8_18"}   # Traffic Light 4
]

# Set up GPIO pins
for light in traffic_lights:
    GPIO.setup(light["red"], GPIO.OUT)
    GPIO.setup(light["yellow"], GPIO.OUT)
    GPIO.setup(light["green"], GPIO.OUT)

def all_off():
    """Turn off all lights"""
    for light in traffic_lights:
        GPIO.output(light["red"], GPIO.LOW)
        GPIO.output(light["yellow"], GPIO.LOW)
        GPIO.output(light["green"], GPIO.LOW)

try:
    print("Starting 4-traffic light sequence. Press Ctrl+C to stop.")

    while True:
        for idx, light in enumerate(traffic_lights):
            print(f"\n🚦 Traffic Light {idx + 1}")

            # Red
            all_off()
            GPIO.output(light["red"], GPIO.HIGH)
            print("RED ON")
            time.sleep(3)

            # Green
            all_off()
            GPIO.output(light["green"], GPIO.HIGH)
            print("GREEN ON")
            time.sleep(3)

            # Yellow
            all_off()
            GPIO.output(light["yellow"], GPIO.HIGH)
            print("YELLOW ON")
            time.sleep(1)

except KeyboardInterrupt:
    print("\nTraffic light sequence stopped by user")
finally:
    all_off()
    GPIO.cleanup()
    print("GPIO pins cleaned up")
