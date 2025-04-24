import Adafruit_BBIO.GPIO as GPIO

# Define pins used in your traffic light setup
pins = ["P8_7", "P8_8", "P8_9", "P8_10", "P8_11", "P8_12", 
        "P8_13", "P8_14", "P8_15", "P8_16", "P8_17", "P8_18"]

# Reset all pins
for pin in pins:
    try:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    except Exception as e:
        print(f"Error resetting pin {pin}: {e}")

# Cleanup
GPIO.cleanup()
print("All pins have been reset and cleaned up")
