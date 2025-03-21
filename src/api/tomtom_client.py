"""
TomTom API Client

This module provides interfaces to interact with TomTom APIs for traffic data collection
at intersections. It handles authentication, request formatting, error handling,
and basic data processing.

Usage:
    client = TomTomClient(api_key='your_api_key')
    traffic_data = client.get_traffic_flow(latitude, longitude)
"""
#this is an example commit message for tutorial purposes

import requests # type: ignore
import json
import time
import logging
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from dotenv import load_dotenv # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TomTomClient:
    """Client for interacting with TomTom APIs to collect traffic data."""
    
    # Base URLs for different TomTom API services
    BASE_URLS = {
        'traffic_flow': 'https://api.tomtom.com/traffic/services/4/flowSegmentData',
        'traffic_incidents': 'https://api.tomtom.com/traffic/services/5/incidentDetails',
        'routing': 'https://api.tomtom.com/routing/1/calculateRoute',
        'snap_to_roads': 'https://api.tomtom.com/traffic/snapToRoads/1/json',
    }
    
    def __init__(self, api_key: str):
        """
        Initialize the TomTom API client.
        
        Args:
            api_key (str): Your TomTom API key
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.max_retries = 3
        self.retry_delay = 1  # seconds
    
    def _make_request(self, url: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make an HTTP request to the TomTom API with improved error handling and retries.
        """
        if params is None:
            params = {}
            
        # Ensure API key is included in the parameters
        if 'key' not in params:
            params['key'] = self.api_key
            
        # Initialize retry counter
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                # Log request details (without API key for security)
                safe_params = {k: v for k, v in params.items() if k != 'key'}
                logger.debug(f"API Request: {url} with params {safe_params}")
                
                response = self.session.get(url, params=params, timeout=10)
                
                # Check if the request was successful
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Too many requests, wait and retry with backoff
                    wait_time = self.retry_delay * (2 ** retries)
                    logger.warning(f"Rate limit hit: {response.text}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                    continue
                elif response.status_code == 400:
                    # Bad request - log details for debugging
                    logger.error(f"Bad request: {response.text}")
                    # Try to return partial data if possible
                    try:
                        return {"error": f"{response.status_code} {response.reason}: {response.text}", 
                                "partial_data": response.json()}
                    except:
                        return {"error": f"{response.status_code} {response.reason}: {response.text}"}
                else:
                    # Log error and raise exception
                    logger.error(f"API Error: Status {response.status_code}, Response: {response.text}")
                    last_error = f"{response.status_code} {response.reason}: {response.text}"
                    retries += 1
                    if retries < self.max_retries:
                        time.sleep(self.retry_delay * (2 ** retries))
                    continue
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                last_error = str(e)
                # Handle network errors with retry
                retries += 1
                if retries < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** retries))
                    continue
                else:
                    return {"error": f"Failed after {self.max_retries} retries: {e}"}
                
        # If we get here, all retries failed
        return {"error": f"Failed after {self.max_retries} retries: {last_error}"}

    def get_traffic_flow(self, latitude: float, longitude: float, zoom: int = 10) -> Dict[str, Any]:
        """
        Get traffic flow data for a specific point.
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            zoom (int, optional): Zoom level for detail (1-20). Defaults to 10.
            
        Returns:
            dict: Traffic flow data
        """
        url = f"{self.BASE_URLS['traffic_flow']}/absolute/{zoom}/json"
        params = {
            'point': f"{latitude},{longitude}",
            'unit': 'KMPH'  # Options: KMPH, MPH
        }
        
        response = self._make_request(url, params)
        logger.info(f"Received traffic flow data for {latitude}, {longitude}")
        return response

    def get_traffic_incidents_in_bbox(self, bbox: Tuple[float, float, float, float]) -> Dict[str, Any]:
        """
        Get traffic incidents within a bounding box.
        
        Args:
            bbox (tuple): Bounding box coordinates (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            dict: Traffic incidents data
        """
        url = f"{self.BASE_URLS['traffic_incidents']}"
        
        # Format bounding box as string: minLon,minLat,maxLon,maxLat
        bbox_str = ','.join(map(str, bbox))
        
        # Simplified fields parameter with proper formatting
        fields_param = '{incidents{type,properties{iconCategory,delay,events{description,code}}}}'
        
        params = {
            'bbox': bbox_str,
            'fields': fields_param,
            'language': 'en-US',
            'categoryFilter': '0,1,2,3,4,5,6,7,8,9,10,11',  # All categories
            'timeValidityFilter': 'present'  
        }
        
        response = self._make_request(url, params)
        logger.info(f"Received traffic incidents data for bbox {bbox_str}")
        return response

    def get_intersection_approaches(self, center_lat: float, center_lon: float, 
                                  radius: float = 0.0005) -> Dict[str, Dict[str, Any]]:
        """
        Get traffic data for all approaches to an intersection by creating points 
        in each cardinal direction from the center.
        
        Args:
            center_lat (float): Center latitude of the intersection
            center_lon (float): Center longitude of the intersection
            radius (float): Distance from center to sample points (in degrees)
            
        Returns:
            dict: Dictionary with traffic data for each approach
        """
        # Define approach points in each cardinal direction
        approaches = {
            "North": (center_lat + radius, center_lon),
            "South": (center_lat - radius, center_lon),
            "East": (center_lat, center_lon + radius),
            "West": (center_lat, center_lon - radius),
            "Northeast": (center_lat + radius * 0.7, center_lon + radius * 0.7),
            "Northwest": (center_lat + radius * 0.7, center_lon - radius * 0.7),
            "Southeast": (center_lat - radius * 0.7, center_lon + radius * 0.7),
            "Southwest": (center_lat - radius * 0.7, center_lon - radius * 0.7)
        }
        
        # Get traffic flow data for each approach
        approach_data = {}
        for direction, (lat, lon) in approaches.items():
            try:
                flow_data = self.get_traffic_flow(lat, lon)
                
                # Verify the data structure
                if 'flowSegmentData' not in flow_data and 'error' not in flow_data:
                    flow_data = {"error": "No flow data returned from API"}
                    
                approach_data[direction] = flow_data
            except Exception as e:
                logger.error(f"Failed to get data for {direction} approach: {e}")
                approach_data[direction] = {"error": str(e)}
                
            # Add a small delay between requests to avoid rate limiting
            time.sleep(0.1)
                    
        return approach_data

    def get_traffic_summary(self, intersection_lat: float, intersection_lon: float) -> Dict[str, Any]:
        """
        Get a comprehensive traffic summary for an intersection including flow data 
        from all approaches and incidents in the vicinity.
        
        Args:
            intersection_lat (float): Latitude of the intersection
            intersection_lon (float): Longitude of the intersection
            
        Returns:
            dict: Traffic summary data
        """
        # Get flow data for all approaches
        approaches_data = self.get_intersection_approaches(intersection_lat, intersection_lon)
        
        # Define a bounding box around the intersection (approximately 300m in each direction)
        # This is roughly 0.003 degrees in latitude/longitude for better performance
        bbox = (
            intersection_lon - 0.003,  # min_lon
            intersection_lat - 0.003,  # min_lat
            intersection_lon + 0.003,  # max_lon
            intersection_lat + 0.003   # max_lat
        )
        
        # Get incident data within the bounding box
        try:
            incidents_data = self.get_traffic_incidents_in_bbox(bbox)
        except Exception as e:
            logger.error(f"Failed to get incident data: {e}")
            incidents_data = {"error": str(e)}
        
        # Compile the summary
        summary = {
            "intersection": {
                "latitude": intersection_lat,
                "longitude": intersection_lon,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            },
            "approaches": {},
            "incidents": incidents_data
        }
        
        # Process and extract relevant data from each approach
        for direction, data in approaches_data.items():
            if "error" in data:
                summary["approaches"][direction] = {"error": data["error"]}
                continue
                
            if "flowSegmentData" in data:
                flow_data = data["flowSegmentData"]
                summary["approaches"][direction] = {
                    "current_speed": flow_data.get("currentSpeed"),
                    "free_flow_speed": flow_data.get("freeFlowSpeed"),
                    "current_travel_time": flow_data.get("currentTravelTime"),
                    "free_flow_travel_time": flow_data.get("freeFlowTravelTime"),
                    "confidence": flow_data.get("confidence"),
                    "road_closure": flow_data.get("roadClosure", False)
                }
                
                # Calculate delay if we have both travel times
                if flow_data.get("currentTravelTime") is not None and flow_data.get("freeFlowTravelTime") is not None:
                    delay = flow_data["currentTravelTime"] - flow_data["freeFlowTravelTime"]
                    summary["approaches"][direction]["delay"] = delay
            else:
                summary["approaches"][direction] = {"error": "No flow data available"}
                
        return summary

    def calculate_route(self, points: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Calculate a route between a series of waypoints.
        
        Args:
            points (list): List of (latitude, longitude) tuples
            
        Returns:
            dict: Routing data
        """
        if len(points) < 2:
            raise ValueError("At least 2 points are required for routing")
            
        # Format waypoints as string: lat1,lon1:lat2,lon2:...
        waypoints = ':'.join([f"{lat},{lon}" for lat, lon in points])
        
        url = f"{self.BASE_URLS['routing']}/{waypoints}/json"
        params = {
            'traffic': 'true',  # Include live traffic data
            'travelMode': 'car',
            'routeType': 'fastest'
        }
        
        response = self._make_request(url, params)
        logger.info(f"Calculated route with {len(points)} waypoints")
        return response
        
    def snap_to_roads(self, points: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Snap a series of GPS points to the road network.
        
        Args:
            points (list): List of (latitude, longitude) tuples
            
        Returns:
            dict: Snapped points data
        """
        url = self.BASE_URLS['snap_to_roads']
        
        # Format the points list for the API
        point_objects = [{"latitude": lat, "longitude": lon} for lat, lon in points]
        
        params = {
            'points': json.dumps(point_objects)
        }
        
        response = self._make_request(url, params)
        logger.info(f"Snapped {len(points)} points to roads")
        return response
def test_api_fixes():
    """
    Test the fixes to the TomTom API integration.
    """
    print("\n=== Testing API Fixes ===")
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment variables
    API_KEY = os.getenv("TOMTOM_API_KEY")
    if not API_KEY:
        print("❌ No API key found. Please set TOMTOM_API_KEY in your .env file.")
        return
    
    # Create a client with the updated code
    client = TomTomClient(api_key=API_KEY)
    
    # North Marietta Pkwy Ne & Cobb Pkwy N intersection
    lat = 33.960192828395996
    lon = -84.52790520126695
    
    # Test traffic incidents with a smaller bounding box
    print("\nTesting Traffic Incidents API with smaller bounding box...")
    bbox = (
        lon - 0.003,  # min_lon
        lat - 0.003,  # min_lat
        lon + 0.003,  # max_lon
        lat + 0.003   # max_lat
    )
    
    incidents_data = client.get_traffic_incidents_in_bbox(bbox)
    if "error" in incidents_data:
        print(f"❌ Error getting incidents: {incidents_data['error']}")
    else:
        print(f"✅ Successfully retrieved incidents data")
        
    # Test traffic summary
    print("\nTesting Traffic Summary...")
    summary = client.get_traffic_summary(lat, lon)
    
    # Check if we got valid data for approaches
    approaches = summary["approaches"]
    success = True
    for direction, data in approaches.items():
        if "error" in data:
            print(f"❌ Error getting data for {direction} approach: {data['error']}")
            success = False
            continue
            
        if "current_speed" not in data:
            print(f"❌ No current_speed found for {direction} approach")
            success = False
            continue
            
        print(f"✅ {direction}: Current Speed={data['current_speed']}kph, Free Flow={data['free_flow_speed']}kph")
    
    # Check incident data
    if "error" in summary["incidents"]:
        print(f"❌ Incidents error: {summary['incidents']['error']}")
    else:
        print(f"✅ Successfully retrieved incidents data in summary")
    
    # Save the summary to a file for inspection
    with open("fixed_traffic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\nFull traffic summary saved to fixed_traffic_summary.json")
    
    return success

# Example usage (will only run if script is executed directly)
if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment variables
    API_KEY = os.getenv("TOMTOM_API_KEY")
    if not API_KEY:
        raise ValueError("No API key found. Please set TOMTOM_API_KEY in your .env file.")
    
    # Create a client
    client = TomTomClient(api_key=API_KEY)
    
    # Run tests
    test_api_fixes()
    
    # Or get a specific traffic summary
    # North Marietta Pkwy Ne & Cobb Pkwy N intersection
    # INTERSECTION_LAT = 33.960192828395996
    # INTERSECTION_LON = -84.52790520126695
    # summary = client.get_traffic_summary(INTERSECTION_LAT, INTERSECTION_LON)
    # print(json.dumps(summary, indent=2))