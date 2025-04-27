# Smart Traffic Control Dashboard

A simple web-based dashboard to monitor your smart traffic control system.

## Overview

This dashboard provides a real-time visual interface to monitor:
- Current status of both intersections
- Traffic light timing plans
- Current congestion levels for each approach
- Recent alerts and system events
- Visual representation of traffic signal states

## Project Structure

```
dashboard/
├── dashboard.py
├── templates/
│   └── index.html
├── certs/  (certificates NOT included)
│   └── [Place your AWS IoT certificates here]
```

> **Note:** The `certs/` folder is intentionally left empty and is listed in `.gitignore` to protect sensitive files. You must provide your own AWS IoT certificates locally.

## Setup Instructions

### 1. Install Required Python Packages

```bash
pip install flask AWSIoTPythonSDK
```

### 2. Obtain AWS IoT Certificates

You will need:
- Device Certificate: `dashboard.pem.crt`
- Private Key: `dashboard.pem.key`
- Root CA Certificate: `AmazonRootCA1.pem`

> **Important:** Do NOT commit your certificates to GitHub.

Place these files inside:

```
dashboard/certs/
```

### 3. Configure the Dashboard

Make sure your `dashboard.py` correctly points to your AWS IoT Core endpoint and uses the certificates from the `certs/` directory.

### 4. Run the Dashboard

```bash
python dashboard.py
```

By default, the dashboard will be available at:

```
http://localhost:8080
```

If running on a BeagleBone or a remote server, access via:

```
http://[your-device-ip]:8080
```

### 5. Command Line Options

- `--port` or `-p`: Specify the port to run on (default: 8080)
- `--host`: Specify the host to bind to (default: 0.0.0.0)
- `--no-sim`: Disable simulated data generation
- `--debug`: Enable Flask debug mode

Example:

```bash
python dashboard.py --port 8888 --debug
```

## Features

- **Real-time Monitoring:** Live updates every 5 seconds
- **Traffic Light Visualization:** North-South and East-West signals
- **Congestion Indicators:** Color-coded traffic levels
- **Alert History:** View recent congestion alerts and system events

## Contributing

Please do not include any certificates, private keys, or sensitive information in your contributions.

## License

This project is licensed for educational and internal use. See LICENSE file for details.

---

**Reminder:** Ensure your AWS IoT certificates stay private and secure. Do not upload them to any public repositories.

