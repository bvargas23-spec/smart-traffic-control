"""
TomTom API Client with Enhanced Capabilities

This module provides interfaces to interact with TomTom APIs for traffic data collection
at intersections, with improved request throttling, response validation, and error handling.

Usage:
    client = TomTomClient(api_key='your_api_key')
    traffic_data = client.get_traffic_flow(latitude, longitude)
"""

import requests
import json
import time
import logging
import os
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom exception classes
class TomTomApiError(Exception):
    """Base exception for TomTom API errors."""
    def __init__(self, message, status_code=None, response=None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(message)

class RateLimitError(TomTomApiError):
    """Exception raised when API rate limits are exceeded."""
    pass

class ValidationError(TomTomApiError):
    """Exception raised when response validation fails."""
    pass

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
        
        # Request settings
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        self.request_timeout = 10  # seconds
        
        # Token bucket for rate limiting
        self.tokens = 10  # Initial tokens (burst capacity)
        self.token_rate = 0.2  # Generate 5 tokens per second (1/5)
        self.last_token_time = time.time()
        self.tokens_lock = threading.RLock()
        
        # Request tracking for rate limits
        self.minute_requests = []
        self.daily_requests = []
        self.max_requests_per_minute = 60
        self.max_requests_per_day = 2500
        
        # Cache for API responses
        self.cache = {}
        self.cache_timeout = 60  # Cache timeout in seconds
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_token_time
        new_tokens = elapsed / self.token_rate
        
        if new_tokens > 0:
            self.tokens = min(10, self.tokens + new_tokens)  # Cap at max burst capacity
            self.last_token_time = now
    
    def _check_rate_limits(self) -> tuple[bool, str]:
        """Check if we're within rate limits."""
        now = time.time()
        
        # Clean up old requests
        minute_ago = now - 60
        self.minute_requests = [t for t in self.minute_requests if t > minute_ago]
        
        day_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        self.daily_requests = [t for t in self.daily_requests if t > day_start]
        
        # Check limits
        if self.tokens < 1:
            return False, "Per-second rate limit reached"
            
        if len(self.minute_requests) >= self.max_requests_per_minute:
            return False, f"Per-minute limit of {self.max_requests_per_minute} requests reached"
            
        if len(self.daily_requests) >= self.max_requests_per_day:
            return False, f"Daily limit of {self.max_requests_per_day} requests reached"
            
        return True, ""
    
    def _throttle_request(self) -> bool:
        """
        Apply throttling to respect rate limits.
        
        Returns:
            bool: True if request can proceed, False if rate limited
        """
        with self.tokens_lock:
            self._refill_tokens()
            can_proceed, reason = self._check_rate_limits()
            
            if can_proceed:
                # Consume a token and track the request
                self.tokens -= 1
                now = time.time()
                self.minute_requests.append(now)
                self.daily_requests.append(now)
                return True
            else:
                logger.warning(f"Rate limit check failed: {reason}")
                return False
    
    def _validate_response(self, response: Dict[str, Any], endpoint_type: str) -> None:
        """
        Validate the structure of the API response.
        
        Args:
            response: The API response to validate
            endpoint_type: Type of endpoint ('flow', 'incidents', etc.)
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(response, dict):
            raise ValidationError(f"Expected dictionary response, got {type(response).__name__}")
            
        # Validate based on endpoint type
        if endpoint_type == 'flow':
            if 'flowSegmentData' not in response:
                raise ValidationError("Missing 'flowSegmentData' in response")
                
            # Check for required fields in flow data
            flow_data = response['flowSegmentData']
            required_fields = ['currentSpeed', 'freeFlowSpeed', 'currentTravelTime', 'freeFlowTravelTime']
            
            for field in required_fields:
                if field not in flow_data:
                    logger.warning(f"Missing field '{field}' in flow data")
                    
        elif endpoint_type == 'incidents':
            if 'incidents' not in response:
                raise ValidationError("Missing 'incidents' in response")
    
    def _make_request(self, url: str, params: Dict[str, Any] = None, 
                    endpoint_type: str = None) -> Dict[str, Any]:
        """
        Make an HTTP request to the TomTom API with improved error handling and retries.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            endpoint_type: Type of endpoint for validation ('flow', 'incidents', etc.)
            
        Returns:
            Parsed JSON response
        """
        if params is None:
            params = {}
            
        # Ensure API key is included in the parameters
        if 'key' not in params:
            params['key'] = self.api_key
            
        # Check cache first
        cache_key = f"{url}_{json.dumps(params, sort_keys=True)}"
        if cache_key in self.cache:
            timestamp, data = self.cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                logger.debug(f"Using cached response for {url}")
                return data
        
        # Try multiple times with backoff
        retries = 0
        while retries <= self.max_retries:
            try:
                # Apply throttling
                if not self._throttle_request():
                    wait_time = 1.0 if retries == 0 else self.retry_delay * (2 ** retries)
                    logger.warning(f"Rate limited. Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                    continue
                
                # Log safe request details (without API key)
                safe_params = {k: v for k, v in params.items() if k != 'key'}
                logger.debug(f"API Request: {url} with params {safe_params}")
                
                # Make the request
                response = self.session.get(url, params=params, timeout=self.request_timeout)
                
                # Check if the request was successful
                if response.status_code == 200:
                    # Parse and validate the response
                    data = response.json()
                    
                    # Validate the response structure if an endpoint type is provided
                    if endpoint_type:
                        try:
                            self._validate_response(data, endpoint_type)
                        except ValidationError as e:
                            logger.warning(f"Response validation warning: {str(e)}")
                    
                    # Cache the response
                    self.cache[cache_key] = (time.time(), data)
                    return data
                    
                elif response.status_code == 429:
                    # Rate limit exceeded, get retry time from header if available
                    retry_after = response.headers.get('Retry-After')
                    wait_time = int(retry_after) if retry_after and retry_after.isdigit() else self.retry_delay * (2 ** retries)
                    
                    logger.warning(f"Rate limit hit (429). Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                    continue
                    
                else:
                    # Handle other error cases
                    error_message = f"HTTP Error {response.status_code}: {response.reason}"
                    
                    # Try to get more details from the response
                    try:
                        error_data = response.json()
                        if isinstance(error_data, dict):
                            error_detail = (
                                error_data.get('errorText') or
                                error_data.get('detailedError', {}).get('message') or
                                error_data.get('error', {}).get('description')
                            )
                            if error_detail:
                                error_message = f"{error_message} - {error_detail}"
                    except (ValueError, KeyError):
                        if response.text:
                            error_message = f"{error_message} - {response.text[:200]}"
                    
                    logger.error(error_message)
                    
                    # Certain error codes should not be retried
                    if response.status_code in [400, 401, 403]:
                        return {"error": error_message}
                    
                    # Otherwise retry
                    retries += 1
                    if retries <= self.max_retries:
                        wait_time = self.retry_delay * (2 ** retries)
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    continue
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request timed out. Retry {retries+1}/{self.max_retries+1}")
                retries += 1
                if retries <= self.max_retries:
                    wait_time = self.retry_delay * (2 ** retries)
                    time.sleep(wait_time)
                continue
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {str(e)}")
                retries += 1
                if retries <= self.max_retries:
                    wait_time = self.retry_delay * (2 ** retries)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                continue
                
        # If we get here, all retries failed
        return {"error": f"Failed after {self.max_retries} retries"}
    
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
        
        response = self._make_request(url, params, endpoint_type='flow')
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
        
        response = self._make_request(url, params, endpoint_type='incidents')
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
                "timestamp": datetime.now().isoformat()
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
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        self.cache = {}
        logger.info("Cache cleared")
    
    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """
        Get current rate limiting statistics.
        
        Returns:
            dict: Current rate limit statistics
        """
        with self.tokens_lock:
            self._refill_tokens()
            
            # Clean up old requests for accurate stats
            now = time.time()
            minute_ago = now - 60
            self.minute_requests = [t for t in self.minute_requests if t > minute_ago]
            
            day_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            self.daily_requests = [t for t in self.daily_requests if t > day_start]
            
            return {
                "tokens_available": self.tokens,
                "minute_requests": len(self.minute_requests),
                "minute_limit": self.max_requests_per_minute,
                "daily_requests": len(self.daily_requests),
                "daily_limit": self.max_requests_per_day
            }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache state.
        
        Returns:
            dict: Cache statistics
        """
        current_time = time.time()
        
        # Count active vs expired cache entries
        active_entries = 0
        expired_entries = 0
        
        for timestamp, _ in self.cache.values():
            if current_time - timestamp < self.cache_timeout:
                active_entries += 1
            else:
                expired_entries += 1
        
        return {
            "total_entries": len(self.cache),
            "active_entries": active_entries,
            "expired_entries": expired_entries,
            "cache_timeout": self.cache_timeout
        }


# Example usage
if __name__ == "__main__":
    # Load API key from environment variable
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    API_KEY = os.getenv("TOMTOM_API_KEY")
    if not API_KEY:
        raise ValueError("No API key found. Please set TOMTOM_API_KEY in your .env file.")
    
    # Create client
    client = TomTomClient(api_key=API_KEY)
    
    try:
        # Example: North Marietta Pkwy Ne & Cobb Pkwy N intersection
        lat = 33.960192828395996
        lon = -84.52790520126695
        
        flow_data = client.get_traffic_flow(lat, lon)
        
        if "error" in flow_data:
            print(f"Error: {flow_data['error']}")
        else:
            segment = flow_data.get("flowSegmentData", {})
            print("\nTraffic Flow Data:")
            print(f"Current Speed: {segment.get('currentSpeed')} kph")
            print(f"Free Flow Speed: {segment.get('freeFlowSpeed')} kph")
            print(f"Current Travel Time: {segment.get('currentTravelTime')} seconds")
            print(f"Confidence: {segment.get('confidence')}")
        
        # Get rate limit stats
        stats = client.get_rate_limit_stats()
        print("\nRate Limit Stats:")
        print(f"Tokens available: {stats['tokens_available']}")
        print(f"Requests in last minute: {stats['minute_requests']}/{stats['minute_limit']}")
        print(f"Requests today: {stats['daily_requests']}/{stats['daily_limit']}")
        
        # Get cache info
        cache_info = client.get_cache_info()
        print("\nCache Info:")
        print(f"Total entries: {cache_info['total_entries']}")
        print(f"Active entries: {cache_info['active_entries']}")
        print(f"Expired entries: {cache_info['expired_entries']}")
        
    except Exception as e:
        print(f"Error: {e}")