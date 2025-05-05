# TomTom API Usage

This document provides guidelines and examples for using the TomTom API in the Smart Traffic Control system.

## API Configuration

### API Key Setup

1. Sign up for a TomTom Developer account at [developer.tomtom.com](https://developer.tomtom.com/)
2. Generate a new API key from your developer dashboard
3. Add the API key to your `.env` file:
   ```
   TOMTOM_API_KEY=your_api_key_here
   ```

## TomTom Client Usage

The project includes a custom TomTom client wrapper (`tomtom_client.py`) that handles authentication, request throttling, and error handling.

### Basic Usage

```python
from src.api.tomtom_client import TomTomClient

# Initialize the client
client = TomTomClient(api_key="your_api_key_here")  # Or leave blank to use from .env

# Get traffic flow data for a specific point
lat = 33.960192  # North Marietta Pkwy and Cobb Pkwy intersection
lon = -84.527905
flow_data = client.get_traffic_flow(lat, lon)

# Get traffic data for all approaches at an intersection
approaches_data = client.get_intersection_approaches(lat, lon)

# Get a complete traffic summary
summary = client.get_traffic_summary(lat, lon)
```

## Available Endpoints

The TomTom client provides access to the following TomTom API endpoints:

### Traffic Flow Segment Data

```python
# Get current traffic flow data
flow_data = client.get_traffic_flow(lat, lon, zoom=10)
```

Returns current speed, free flow speed, travel times, and confidence values for the road segment.

### Traffic Incidents

```python
# Define a bounding box around an area
bbox = (min_lon, min_lat, max_lon, max_lat)

# Get traffic incidents in the area
incidents = client.get_traffic_incidents_in_bbox(bbox)
```

Returns accidents, construction, and other incidents affecting traffic in the specified area.

### Intersection Approaches

```python
# Get data for all approaches to an intersection
approaches = client.get_intersection_approaches(center_lat, center_lon, radius=0.0005)
```

Creates points in each cardinal direction from the center and fetches traffic data for each approach.

## Data Processing

After retrieving data from the TomTom API, use the `DataProcessor` class to transform raw data into actionable metrics:

```python
from src.api.data_processor import DataProcessor

# Initialize the processor
processor = DataProcessor()

# Process traffic flow data
processed_flow = processor.process_traffic_flow(flow_data)

# Process intersection data
processed_intersection = processor.process_intersection_data({
    "intersection": summary["intersection"],
    "approaches": summary["approaches"]
})

# Calculate optimal traffic signal timing
timing_recommendations = processor.calculate_optimal_cycle_times(processed_intersection)
```

## Rate Limits and Best Practices

- The free TomTom API plan includes 2,500 daily transactions
- Implement appropriate caching to minimize redundant requests
- Avoid polling too frequently (once per minute is usually sufficient)
- Handle API errors gracefully with retry logic

## Error Handling

The TomTom client includes built-in error handling:

```python
try:
    data = client.get_traffic_flow(lat, lon)
    if "error" in data:
        print(f"Error fetching traffic data: {data['error']}")
    else:
        # Process the data
        processed_data = processor.process_traffic_flow(data)
except Exception as e:
    print(f"Exception: {e}")
```

## Advanced Features

### Geographic Approaches

For intersection analysis, the system uses cardinal directions to model approaches:

```python
# Define intersection with approaches
intersection_points = {
    "North": (33.95177778, -84.52108333),
    "South": (33.95030556, -84.52027778),
    "East": (33.95091667, -84.51977778),
    "West": (33.95097222, -84.52163889)
}

# Get traffic data for each approach
traffic_data = client.get_intersection_traffic(intersection_points)
```

### Customizing Request Parameters

Advanced users can customize request parameters:

```python
# Get traffic flow with custom parameters
custom_flow_data = client.get_flow_data(
    lat=33.960192,
    lon=-84.527905,
    style="relative",  # Use relative speeds
    zoom=15,           # Higher detail level
    format="json"
)
```

## Troubleshooting

If you encounter issues with the TomTom API:

1. Verify your API key is valid and has sufficient quota
2. Check your internet connection
3. Ensure coordinates are within valid ranges
4. Check the API response for error messages
5. Look at the TomTom client logs for request/response details

For additional help, refer to the [TomTom Developer Portal](https://developer.tomtom.com/)