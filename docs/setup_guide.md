# Hardware Setup Guide

This guide provides detailed instructions for setting up the hardware components of the Smart Traffic Control System.

## Hardware Requirements

### Core Components

- BeagleBone Black (Rev C or newer)
- MicroSD card (8GB or larger)
- 5V 2A power supply with barrel jack connector
- Ethernet cable or WiFi adapter
- USB to TTL serial adapter (for initial setup)

### Traffic Light Control Components

- 12-16 LEDs (red, yellow, green for each approach)
- Resistors (220Ω-330Ω for LEDs)
- Transistors (2N2222 or similar, one per LED)
- Prototype board (breadboard for testing, PCB for permanent installation)
- Jumper wires
- DIN rail mountable enclosure (for permanent installation)

### Optional Components

- 12V power supply (if using actual traffic light modules)
- Relay board (if controlling higher voltage devices)
- Weather-resistant enclosure (for outdoor deployment)

## BeagleBone Black Setup

### Operating System Installation

1. Download the latest Debian image for BeagleBone Black from [beagleboard.org](https://beagleboard.org/latest-images)

2. Write the image to a microSD card using Balena Etcher or similar tool:
   ```bash
   sudo dd if=bone-debian-10.3-iot-armhf-2020-04-06-4gb.img of=/dev/mmcblk0 bs=1M
   ```

3. Insert the microSD card into the BeagleBone Black

4. Connect power while holding the boot button to boot from the microSD card

5. Connect to the BeagleBone via USB or network:
   ```bash
   ssh debian@192.168.7.2  # Default password: temppwd
   ```

6. Update the system:
   ```bash
   sudo apt update
   sudo apt upgrade
   ```

### Network Configuration

#### Ethernet Connection

1. Connect an Ethernet cable to the BeagleBone Black

2. Edit the network configuration:
   ```bash
   sudo nano /etc/network/interfaces
   ```

3. Configure for DHCP:
   ```
   auto eth0
   iface eth0 inet dhcp
   ```

4. Or configure a static IP:
   ```
   auto eth0
   iface eth0 inet static
       address 192.168.1.100
       netmask 255.255.255.0
       gateway 192.168.1.1
   ```

5. Restart networking:
   ```bash
   sudo systemctl restart networking
   ```

#### WiFi Connection (with USB adapter)

1. Install wifi tools:
   ```bash
   sudo apt install wireless-tools wpasupplicant
   ```

2. Scan for networks:
   ```bash
   sudo iwlist wlan0 scan | grep ESSID
   ```

3. Configure WiFi:
   ```bash
   sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
   ```

4. Add network details:
   ```
   network={
       ssid="YourNetworkName"
       psk="YourPassword"
   }
   ```

5. Enable on boot:
   ```bash
   sudo systemctl enable wpa_supplicant
   ```

## GPIO Configuration

### Enable Required Device Tree Overlays

1. Edit the uEnv.txt file:
   ```bash
   sudo nano /boot/uEnv.txt
   ```

2. Uncomment the following line:
   ```
   enable_uboot_overlays=1
   ```

3. Reboot the BeagleBone:
   ```bash
   sudo reboot
   ```

### Install Required Libraries

1. Install Python and GPIO libraries:
   ```bash
   sudo apt install python3 python3-pip
   sudo pip3 install Adafruit-BBIO
   ```

2. Test GPIO functionality:
   ```bash
   python3 -c "import Adafruit_BBIO.GPIO as GPIO; print('GPIO module loaded successfully')"
   ```

## Traffic Light Wiring

### Basic LED Circuit

For each traffic light LED:

1. Connect a GPIO pin to a 220Ω resistor
2. Connect the resistor to the LED anode (longer leg)
3. Connect the LED cathode (shorter leg) to ground

### Transistor-Based Circuit (Recommended)

For more reliable operation, use transistors to switch the LEDs:

1. Connect a GPIO pin through a 1kΩ resistor to the transistor base
2. Connect the transistor collector to the LED anode through a current-limiting resistor
3. Connect the LED cathode to the transistor emitter
4. Connect the emitter to ground

### Standard Pin Mapping

Use the following GPIO pin mapping for consistency:

```
North Traffic Light:
- Red:    P8_7
- Yellow: P8_8
- Green:  P8_9

South Traffic Light:
- Red:    P8_10
- Yellow: P8_11
- Green:  P8_12

East Traffic Light:
- Red:    P8_13
- Yellow: P8_14
- Green:  P8_15

West Traffic Light:
- Red:    P8_16
- Yellow: P8_17
- Green:  P8_18
```

## Circuit Diagram

```
BeagleBone GPIO Pin (P8_X) --> 1kΩ Resistor --> Transistor Base
                                                    |
                                                Collector
                                                    |
                                               220Ω Resistor 
                                                    |
                                                LED Anode
                                                    |
                                                LED Cathode
                                                    |
                                               Transistor Emitter
                                                    |
                                                  Ground
```

## Software Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bvargas23-spec/smart-traffic-control.git
   cd smart-traffic-control
   ```

2. Install requirements:
   ```bash
   pip3 install -r requirements.txt
   ```

3. Configure environment:
   ```bash
   cp .env.example .env
   nano .env
   # Add your TomTom API key
   ```

4. Test the system with the provided script:
   ```bash
   python3 tests/test_lights.py
   ```

## Troubleshooting

### Common GPIO Issues

1. **Permission denied error**:
   ```bash
   sudo chmod a+rw /sys/class/gpio/*
   ```

2. **GPIO pins not working**:
   - Ensure dtb overlays are enabled in uEnv.txt
   - Check that pins aren't being used by other services
   - Verify physical connections

3. **Hardware resources already in use**:
   ```bash
   sudo systemctl stop bonescript.service
   sudo systemctl disable bonescript.service
   ```

### Hardware Verification

Use the included test script to verify hardware connections:

```bash
python3 src/controllers/reset_pins.py  # Reset all pins
python3 tests/intersection_test.py     # Run a simple traffic light sequence
```

## System Startup Configuration

To run the system on boot:

1. Create a systemd service:
   ```bash
   sudo nano /etc/systemd/system/traffic-control.service
   ```

2. Add service configuration:
   ```
   [Unit]
   Description=Smart Traffic Control System
   After=network.target

   [Service]
   Type=simple
   User=debian
   WorkingDirectory=/home/debian/smart-traffic-control
   ExecStart=/usr/bin/python3 src/main.py
   Restart=on-failure
   RestartSec=5s

   [Install]
   WantedBy=multi-user.target
   ```

3. Enable and start the service:
   ```bash
   sudo systemctl enable traffic-control.service
   sudo systemctl start traffic-control.service
   ```

4. Check service status:
   ```bash
   sudo systemctl status traffic-control.service
   ```

## Physical Installation Notes

When deploying the system to control actual traffic lights:

1. Use weatherproof enclosures rated for outdoor use
2. Implement proper electrical isolation between control circuitry and traffic signals
3. Follow local electrical codes for traffic control equipment
4. Consider using optical isolation for safety in high-voltage applications
5. Install surge protection for both power and signal lines

## Maintenance

1. Regularly check physical connections for corrosion or damage
2. Monitor system logs for unusual patterns:
   ```bash
   sudo journalctl -u traffic-control -f
   ```
3. Update the software periodically:
   ```bash
   cd smart-traffic-control
   git pull
   pip3 install -r requirements.txt
   ```
4. Back up your configuration files before making changes

## Advanced Configuration

For multi-intersection deployments or custom hardware setups, refer to the documentation in `examples/` for detailed examples and configuration options.