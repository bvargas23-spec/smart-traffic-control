import os
import time
import json
import pandas as pd # type: ignore
import folium # type: ignore
import numpy as np # type: ignore
from datetime import datetime
import requests # type: ignore
from dotenv import load_dotenv # type: ignore

# Load environment variables from .env file
load_dotenv()

class TomTomClient:
    """Client for interacting with TomTom Traffic APIs"""
    
    def __init__(self, api_key=None):
        """Initialize the TomTom API client"""
        self.api_key = api_key or os.getenv("TOMTOM_API_KEY")
        if not self.api_key:
            raise ValueError("No API key found. Set TOMTOM_API_KEY in .env or pass as parameter.")
        
        # Base URLs for different TomTom APIs
        self.flow_base_url = "https://api.tomtom.com/traffic/services/4/flowSegmentData"
        self.incidents_base_url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
        self.routing_base_url = "https://api.tomtom.com/routing/1/calculateRoute"
        
        # Cache for API responses to avoid redundant calls
        self.cache = {}
        self.cache_timeout = 60  # Cache timeout in seconds

    def get_flow_data(self, lat, lon, style="absolute", zoom=10, format="json"):
        """Get traffic flow data for a specific point
        
        Args:
            lat: Latitude of the point
            lon: Longitude of the point
            style: Flow data style (absolute, relative, relative0)
            zoom: Zoom level (affects road network detail)
            format: Response format (json)
            
        Returns:
            Dict containing flow data or None on error
        """
        cache_key = f"flow_{lat}_{lon}_{style}_{zoom}"
        
        # Check cache first
        if cache_key in self.cache:
            timestamp, data = self.cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return data
        
        # Construct the flow API URL
        url = f"{self.flow_base_url}/{style}/{zoom}/{format}"
        params = {
            "point": f"{lat},{lon}",
            "unit": "KMPH",
            "key": self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()
            
            # Cache the response
            self.cache[cache_key] = (time.time(), data)
            
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching flow data: {e}")
            return None

    def get_incidents_data(self, bbox, fields=None, language="en-US"):
        """Get traffic incidents within a bounding box
        
        Args:
            bbox: Bounding box as string "minLon,minLat,maxLon,maxLat"
            fields: Fields to include in the response
            language: Language for incident descriptions
            
        Returns:
            Dict containing incidents data or None on error
        """
        cache_key = f"incidents_{bbox}"
        
        # Check cache first
        if cache_key in self.cache:
            timestamp, data = self.cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return data
        
        # Default fields if not specified
        if not fields:
            fields = "{incidents{type,geometry{type,coordinates},properties{iconCategory,description,delay,events{description,code},startTime,endTime,from,to}}}"
        
        # Get current timestamp for t parameter
        timestamp = int(time.time())
        
        params = {
            "bbox": bbox,
            "fields": fields,
            "language": language,
            "categoryFilter": "0,1,2,3,4,5,6,7,8,9,10,11",
            "timeValidityFilter": "present",
            "key": self.api_key
        }
        
        try:
            response = requests.get(self.incidents_base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Cache the response
            self.cache[cache_key] = (time.time(), data)
            
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching incidents data: {e}")
            return None
            
    def calculate_route(self, waypoints, traffic=True, route_type="fastest", travel_mode="car"):
        """Calculate a route between multiple waypoints
        
        Args:
            waypoints: List of (lat, lon) tuples defining the route
            traffic: Consider traffic conditions
            route_type: Type of route (fastest, shortest, eco)
            travel_mode: Mode of transportation (car, truck, pedestrian)
            
        Returns:
            Dict containing route data or None on error
        """
        # Convert waypoints to string format: lat1,lon1:lat2,lon2:...
        waypoints_str = ":".join([f"{lat},{lon}" for lat, lon in waypoints])
        
        cache_key = f"route_{waypoints_str}_{traffic}_{route_type}_{travel_mode}"
        
        # Check cache first
        if cache_key in self.cache:
            timestamp, data = self.cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return data
        
        # Construct the URL and parameters
        url = f"{self.routing_base_url}/{waypoints_str}/json"
        params = {
            "key": self.api_key,
            "routeType": route_type,
            "traffic": str(traffic).lower(),
            "travelMode": travel_mode
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Cache the response
            self.cache[cache_key] = (time.time(), data)
            
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error calculating route: {e}")
            return None


class TrafficDataProcessor:
    """Process and analyze traffic data from TomTom API"""
    
    def __init__(self, tomtom_client):
        """Initialize with a TomTom client"""
        self.tomtom_client = tomtom_client
        self.data_history = {}  # Store historical data
    
    def get_point_traffic(self, lat, lon, point_name=None):
        """Get processed traffic data for a specific point
        
        Args:
            lat: Latitude of the point
            lon: Longitude of the point
            point_name: Optional name for the point
            
        Returns:
            Dict with processed traffic metrics
        """
        flow_data = self.tomtom_client.get_flow_data(lat, lon)
        
        if not flow_data or "flowSegmentData" not in flow_data:
            print(f"No flow data available for point {lat}, {lon}")
            return None
        
        segment = flow_data["flowSegmentData"]
        
        # Extract base metrics
        current_speed = segment.get("currentSpeed", 0)
        free_flow_speed = segment.get("freeFlowSpeed", 0)
        current_tt = segment.get("currentTravelTime", 0)
        free_flow_tt = segment.get("freeFlowTravelTime", 0)
        confidence = segment.get("confidence", 0)
        road_closure = segment.get("roadClosure", False)
        
        # Calculate derived metrics
        delay = current_tt - free_flow_tt if current_tt and free_flow_tt else 0
        speed_ratio = current_speed / free_flow_speed if free_flow_speed else 0
        
        # Calculate congestion level (0-5 scale)
        if road_closure:
            congestion_level = 5
        elif speed_ratio >= 0.9:
            congestion_level = 0  # Free flowing
        elif speed_ratio >= 0.7:
            congestion_level = 1  # Light
        elif speed_ratio >= 0.5:
            congestion_level = 2  # Moderate
        elif speed_ratio >= 0.3:
            congestion_level = 3  # Heavy
        elif speed_ratio > 0:
            congestion_level = 4  # Very heavy
        else:
            congestion_level = 5  # Stopped/Blocked
            
        # Calculate traffic score (0-100, lower is worse)
        if road_closure:
            traffic_score = 0
        else:
            # Base score on speed ratio with confidence factor
            traffic_score = min(100, max(0, round(speed_ratio * 100 * confidence)))
        
        # Create result dictionary
        result = {
            "lat": lat,
            "lon": lon,
            "name": point_name,
            "timestamp": datetime.now().isoformat(),
            "current_speed": current_speed,
            "free_flow_speed": free_flow_speed,
            "current_travel_time": current_tt,
            "free_flow_travel_time": free_flow_tt,
            "delay": delay,
            "confidence": confidence,
            "road_closure": road_closure,
            "speed_ratio": speed_ratio,
            "congestion_level": congestion_level,
            "traffic_score": traffic_score
        }
        
        # Store in history
        point_key = point_name or f"{lat},{lon}"
        if point_key not in self.data_history:
            self.data_history[point_key] = []
        self.data_history[point_key].append(result)
        
        # Trim history if it gets too large
        if len(self.data_history[point_key]) > 100:
            self.data_history[point_key] = self.data_history[point_key][-100:]
            
        return result
    
    def get_intersection_traffic(self, intersection_points):
        """Get traffic data for all approaches at an intersection
        
        Args:
            intersection_points: Dict mapping approach names to (lat, lon) tuples
            
        Returns:
            Dict mapping approach names to traffic data
        """
        results = {}
        for name, (lat, lon) in intersection_points.items():
            data = self.get_point_traffic(lat, lon, name)
            if data:
                results[name] = data
        return results
    
    def get_incidents_in_area(self, min_lat, min_lon, max_lat, max_lon):
        """Get traffic incidents in the specified bounding box
        
        Args:
            min_lat: Minimum latitude of bounding box
            min_lon: Minimum longitude of bounding box
            max_lat: Maximum latitude of bounding box
            max_lon: Maximum longitude of bounding box
            
        Returns:
            List of processed incident data
        """
        bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
        incidents_data = self.tomtom_client.get_incidents_data(bbox)
        
        if not incidents_data or "incidents" not in incidents_data:
            return []
            
        processed_incidents = []
        for incident in incidents_data["incidents"]:
            # Extract properties
            props = incident.get("properties", {})
            
            # Get coordinates (try to extract from geometry)
            coords = None
            geometry = incident.get("geometry", {})
            if geometry and geometry.get("type") == "LineString":
                # Take midpoint of the LineString for display
                line_coords = geometry.get("coordinates", [])
                if line_coords and len(line_coords) > 0:
                    # If it's just one point, use it
                    if len(line_coords) == 1:
                        coords = line_coords[0]
                    else:
                        # Use the midpoint
                        mid_idx = len(line_coords) // 2
                        coords = line_coords[mid_idx]
            
            # If no coordinates found, skip this incident
            if not coords:
                continue
                
            # Process the incident
            processed = {
                "type": incident.get("type", "Unknown"),
                "lon": coords[0],
                "lat": coords[1],
                "description": props.get("description", "No description"),
                "delay": props.get("delay", 0),
                "from": props.get("from", ""),
                "to": props.get("to", ""),
                "icon_category": props.get("iconCategory", 0),
                "start_time": props.get("startTime", ""),
                "end_time": props.get("endTime", "")
            }
            
            processed_incidents.append(processed)
            
        return processed_incidents
    
    def save_data_to_csv(self, filepath):
        """Save all historical data to a CSV file
        
        Args:
            filepath: Path to the CSV file to save
        """
        all_data = []
        for point_key, history in self.data_history.items():
            all_data.extend(history)
            
        if not all_data:
            print("No data to save")
            return
            
        # Convert to DataFrame and save
        df = pd.DataFrame(all_data)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")


class TrafficVisualizer:
    """Create visualizations of traffic data"""
    
    def __init__(self, data_processor):
        """Initialize with a data processor"""
        self.data_processor = data_processor
        
    def get_congestion_color(self, congestion_level):
        """Return color for congestion level
        
        Args:
            congestion_level: Integer 0-5 congestion level
            
        Returns:
            Color string for the map
        """
        colors = {
            0: "green",      # Free flowing
            1: "lightgreen", # Light
            2: "yellow",     # Moderate
            3: "orange",     # Heavy
            4: "red",        # Very heavy
            5: "darkred"     # Stopped/Blocked
        }
        return colors.get(congestion_level, "gray")
    
    def get_marker_icon(self, traffic_data):
        """Create folium icon based on traffic data
        
        Args:
            traffic_data: Dict with processed traffic data
            
        Returns:
            folium.Icon object
        """
        congestion_level = traffic_data.get("congestion_level", 0)
        color = self.get_congestion_color(congestion_level)
        
        return folium.Icon(
            color=color,
            icon="info-sign",
            prefix="glyphicon"
        )
    
    def create_traffic_map(self, intersection_points, center_lat=None, center_lon=None, zoom_start=15):
        """Create a folium map with traffic data for an intersection
        
        Args:
            intersection_points: Dict mapping approach names to (lat, lon) tuples
            center_lat: Center latitude for the map (if None, calculated)
            center_lon: Center longitude for the map (if None, calculated)
            zoom_start: Initial zoom level
            
        Returns:
            folium.Map object
        """
        # Get traffic data for the intersection
        traffic_data = self.data_processor.get_intersection_traffic(intersection_points)
        
        # Calculate center if not provided
        if center_lat is None or center_lon is None:
            lats = [point[0] for point in intersection_points.values()]
            lons = [point[1] for point in intersection_points.values()]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
        
        # Create the map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
        
        # Calculate bounding box (with some padding)
        lats = [point[0] for point in intersection_points.values()]
        lons = [point[1] for point in intersection_points.values()]
        min_lat = min(lats) - 0.001
        max_lat = max(lats) + 0.001
        min_lon = min(lons) - 0.001
        max_lon = max(lons) + 0.001
        
        # Get incidents in the area
        incidents = self.data_processor.get_incidents_in_area(
            min_lat, min_lon, max_lat, max_lon
        )
        
        # Add traffic data markers
        for name, data in traffic_data.items():
            if not data:
                continue
                
            # Create popup content
            popup_content = f"""
            <b>{name}</b><br>
            Current Speed: {data['current_speed']} kph<br>
            Free Flow Speed: {data['free_flow_speed']} kph<br>
            Current TT: {data['current_travel_time']} s<br>
            Free Flow TT: {data['free_flow_travel_time']} s<br>
            Delay: {data['delay']} s<br>
            Congestion Level: {data['congestion_level']}/5<br>
            Traffic Score: {data['traffic_score']}/100<br>
            Confidence: {data['confidence']}
            """
            
            # Add marker
            folium.Marker(
                location=[data['lat'], data['lon']],
                popup=popup_content,
                tooltip=f"{name}: Level {data['congestion_level']}",
                icon=self.get_marker_icon(data)
            ).add_to(m)
        
        # Add incident markers
        for incident in incidents:
            popup_content = f"""
            <b>Incident: {incident['type']}</b><br>
            {incident['description']}<br>
            Delay: {incident['delay']} s<br>
            From: {incident['from']}<br>
            To: {incident['to']}
            """
            
            folium.Marker(
                location=[incident['lat'], incident['lon']],
                popup=popup_content,
                tooltip=f"Traffic Incident",
                icon=folium.Icon(color="purple", icon="warning-sign", prefix="glyphicon")
            ).add_to(m)
        
        # Add the bounding box
        folium.Rectangle(
            bounds=[(min_lat, min_lon), (max_lat, max_lon)],
            color="blue",
            fill=True,
            fill_opacity=0.2,
            popup="Traffic Data Area"
        ).add_to(m)
        
        # Add a legend (custom HTML)
        legend_html = """
        <div style="position: fixed; 
             bottom: 50px; right: 50px; width: 150px; height: 160px; 
             border:2px solid grey; z-index:9999; font-size:12px;
             background-color: white; padding: 10px;
             border-radius: 5px;">
             <p><b>Congestion Levels</b></p>
             <p><i class="fa fa-circle" style="color:green"></i> Free Flowing (0)</p>
             <p><i class="fa fa-circle" style="color:lightgreen"></i> Light (1)</p>
             <p><i class="fa fa-circle" style="color:yellow"></i> Moderate (2)</p>
             <p><i class="fa fa-circle" style="color:orange"></i> Heavy (3)</p>
             <p><i class="fa fa-circle" style="color:red"></i> Very Heavy (4)</p>
             <p><i class="fa fa-circle" style="color:darkred"></i> Stopped (5)</p>
             <p><i class="fa fa-exclamation-triangle" style="color:purple"></i> Incident</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add a timestamp
        timestamp_html = f"""
        <div style="position: fixed; 
             top: 10px; right: 10px; width: auto; 
             border:2px solid grey; z-index:9999; font-size:12px;
             background-color: white; padding: 5px;
             border-radius: 5px;">
             Data as of: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
        """
        m.get_root().html.add_child(folium.Element(timestamp_html))
        
        return m
    
    def create_multi_intersection_map(self, intersections, center_lat=None, center_lon=None, zoom_start=13):
        """Create a map with multiple intersections
        
        Args:
            intersections: Dict mapping intersection names to dicts of points
            center_lat: Center latitude for the map
            center_lon: Center longitude for the map
            zoom_start: Initial zoom level
            
        Returns:
            folium.Map object
        """
        # Calculate center if not provided
        if center_lat is None or center_lon is None:
            all_lats = []
            all_lons = []
            for intersection_points in intersections.values():
                all_lats.extend([point[0] for point in intersection_points.values()])
                all_lons.extend([point[1] for point in intersection_points.values()])
            center_lat = sum(all_lats) / len(all_lats) if all_lats else 0
            center_lon = sum(all_lons) / len(all_lons) if all_lons else 0
        
        # Create the map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
        
        # Calculate overall bounding box
        all_lats = []
        all_lons = []
        for intersection_points in intersections.values():
            all_lats.extend([point[0] for point in intersection_points.values()])
            all_lons.extend([point[1] for point in intersection_points.values()])
        
        min_lat = min(all_lats) - 0.002
        max_lat = max(all_lats) + 0.002
        min_lon = min(all_lons) - 0.002
        max_lon = max(all_lons) + 0.002
        
        # Get incidents in the area
        incidents = self.data_processor.get_incidents_in_area(
            min_lat, min_lon, max_lat, max_lon
        )
        
        # Process each intersection
        for intersection_name, intersection_points in intersections.items():
            # Get traffic data
            traffic_data = self.data_processor.get_intersection_traffic(intersection_points)
            
            # Calculate center of this intersection
            int_lats = [point[0] for point in intersection_points.values()]
            int_lons = [point[1] for point in intersection_points.values()]
            int_center_lat = sum(int_lats) / len(int_lats)
            int_center_lon = sum(int_lons) / len(int_lons)
            
            # Add a circle marker for the intersection center
            folium.CircleMarker(
                location=[int_center_lat, int_center_lon],
                radius=5,
                popup=intersection_name,
                color="blue",
                fill=True,
                fill_color="blue"
            ).add_to(m)
            
            # Add traffic data markers
            for name, data in traffic_data.items():
                if not data:
                    continue
                    
                # Create popup content with intersection name
                popup_content = f"""
                <b>{intersection_name}: {name}</b><br>
                Current Speed: {data['current_speed']} kph<br>
                Free Flow Speed: {data['free_flow_speed']} kph<br>
                Current TT: {data['current_travel_time']} s<br>
                Free Flow TT: {data['free_flow_travel_time']} s<br>
                Delay: {data['delay']} s<br>
                Congestion Level: {data['congestion_level']}/5<br>
                Traffic Score: {data['traffic_score']}/100<br>
                Confidence: {data['confidence']}
                """
                
                # Add marker
                folium.Marker(
                    location=[data['lat'], data['lon']],
                    popup=popup_content,
                    tooltip=f"{intersection_name}: {name}",
                    icon=self.get_marker_icon(data)
                ).add_to(m)
        
        # Add incident markers
        for incident in incidents:
            popup_content = f"""
            <b>Incident: {incident['type']}</b><br>
            {incident['description']}<br>
            Delay: {incident['delay']} s<br>
            From: {incident['from']}<br>
            To: {incident['to']}
            """
            
            folium.Marker(
                location=[incident['lat'], incident['lon']],
                popup=popup_content,
                tooltip=f"Traffic Incident",
                icon=folium.Icon(color="purple", icon="warning-sign", prefix="glyphicon")
            ).add_to(m)
        
        # Add the bounding box
        folium.Rectangle(
            bounds=[(min_lat, min_lon), (max_lat, max_lon)],
            color="blue",
            fill=True,
            fill_opacity=0.1,
            popup="Traffic Monitoring Area"
        ).add_to(m)
        
        # Add a legend
        legend_html = """
        <div style="position: fixed; 
             bottom: 50px; right: 50px; width: 150px; height: 160px; 
             border:2px solid grey; z-index:9999; font-size:12px;
             background-color: white; padding: 10px;
             border-radius: 5px;">
             <p><b>Congestion Levels</b></p>
             <p><i class="fa fa-circle" style="color:green"></i> Free Flowing (0)</p>
             <p><i class="fa fa-circle" style="color:lightgreen"></i> Light (1)</p>
             <p><i class="fa fa-circle" style="color:yellow"></i> Moderate (2)</p>
             <p><i class="fa fa-circle" style="color:orange"></i> Heavy (3)</p>
             <p><i class="fa fa-circle" style="color:red"></i> Very Heavy (4)</p>
             <p><i class="fa fa-circle" style="color:darkred"></i> Stopped (5)</p>
             <p><i class="fa fa-exclamation-triangle" style="color:purple"></i> Incident</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add a timestamp
        timestamp_html = f"""
        <div style="position: fixed; 
             top: 10px; right: 10px; width: auto; 
             border:2px solid grey; z-index:9999; font-size:12px;
             background-color: white; padding: 5px;
             border-radius: 5px;">
             Data as of: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
        """
        m.get_root().html.add_child(folium.Element(timestamp_html))
        
        return m


def demo_single_intersection():
    """Demo showing traffic monitoring for a single intersection"""
    # Initialize client and data processor
    tomtom_client = TomTomClient()
    data_processor = TrafficDataProcessor(tomtom_client)
    visualizer = TrafficVisualizer(data_processor)
    
    # Define intersection with points for each approach
    # Example: Roswell Street NE & Hwy 41 intersection
    intersection_points = {
        "North Approach": (33.95177778, -84.52108333),
        "South Approach": (33.95030556, -84.52027778),
        "East Approach": (33.95091667, -84.51977778),
        "West Approach": (33.95097222, -84.52163889)
    }
    
    # Create the map
    m = visualizer.create_traffic_map(intersection_points, zoom_start=18)
    
    # Save the map
    output_file = "intersection_traffic_map.html"
    m.save(output_file)
    print(f"Map saved to {output_file}")
    
    # Save data to CSV
    data_processor.save_data_to_csv("intersection_traffic_data.csv")


def demo_multi_intersection():
    """Demo showing traffic monitoring for multiple intersections"""
    # Initialize client and data processor
    tomtom_client = TomTomClient()
    data_processor = TrafficDataProcessor(tomtom_client)
    visualizer = TrafficVisualizer(data_processor)
    
    # Define multiple intersections
    intersections = {
        "Roswell & Hwy 41": {  # First intersection
            "North": (33.95177778, -84.52108333),
            "South": (33.95030556, -84.52027778),
            "East": (33.95091667, -84.51977778),
            "West": (33.95097222, -84.52163889)
        },
        "Cobb Pkwy & Hwy 120": {
            "North": (33.94230, -84.51568),  # North approach on Cobb Parkway
            "South": (33.94125, -84.51568),  # South approach on Cobb Parkway
            "East": (33.94178, -84.51470),   # East approach on S Marietta Pkwy
            "West": (33.94178, -84.51670)    # West approach on S Marietta Pkwy
        }
    }
    
    # Create the map
    m = visualizer.create_multi_intersection_map(intersections, zoom_start=14)
    
    # Save the map
    output_file = "multi_intersection_traffic_map.html"
    m.save(output_file)
    print(f"Map saved to {output_file}")
    
    # Save data to CSV
    data_processor.save_data_to_csv("multi_intersection_traffic_data.csv")


def continuous_monitoring(interval=300):
    """Run continuous monitoring of traffic with periodic updates
    
    Args:
        interval: Update interval in seconds (default 5 minutes)
    """
    # Initialize client and data processor
    tomtom_client = TomTomClient()
    data_processor = TrafficDataProcessor(tomtom_client)
    visualizer = TrafficVisualizer(data_processor)
    
    # Define intersection with points for each approach
    # Example: Roswell Street NE & Hwy 41 intersection
    intersection_points = {
        "North Approach": (33.95177778, -84.52108333),
        "South Approach": (33.95030556, -84.52027778),
        "East Approach": (33.95091667, -84.51977778),
        "West Approach": (33.95097222, -84.52163889)
    }
    
    print(f"Starting continuous traffic monitoring (update every {interval} seconds)")
    print("Press Ctrl+C to stop")
    
    try:
        count = 0
        while True:
            count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"\nUpdate #{count} at {timestamp}")
            
            # Create the map
            m = visualizer.create_traffic_map(intersection_points, zoom_start=18)
            
            # Save the map with timestamp
            output_file = f"traffic_map_{timestamp}.html"
            m.save(output_file)
            print(f"Map saved to {output_file}")
            
            # Also save a copy as latest.html
            latest_file = "traffic_map_latest.html"
            m.save(latest_file)
            print(f"Latest map saved to {latest_file}")
            
            # Save data to CSV
            data_processor.save_data_to_csv("traffic_data.csv")
            
            # Wait for next update
            print(f"Waiting {interval} seconds until next update...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"\nError during monitoring: {e}")
    finally:
        # Save final data export
        data_processor.save_data_to_csv("traffic_data_final.csv")
        print("Final data saved to traffic_data_final.csv")


if __name__ == "__main__":
    # Choose one of the demo functions to run
    print("TomTom Traffic Visualization System")
    print("1. Single Intersection Demo")
    print("2. Multi-Intersection Demo")
    print("3. Continuous Monitoring")
    
    choice = input("Select an option (1-3): ")
    
    if choice == "1":
        demo_single_intersection()
    elif choice == "2":
        demo_multi_intersection()
    elif choice == "3":
        interval = input("Enter update interval in seconds (default 300): ")
        try:
            interval = int(interval) if interval.strip() else 300
        except ValueError:
            print("Invalid interval, using default 300 seconds")
            interval = 300
        continuous_monitoring(interval)
    else:
        print("Invalid choice, running single intersection demo")
        demo_single_intersection()