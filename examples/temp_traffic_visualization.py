# traffic_visualization.py
import folium # type: ignore
import pandas as pd # type: ignore
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv # type: ignore
import traceback
import math

# Add the project root to the path so we can import modules correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import our modules
from src.api.tomtom_client import TomTomClient
from src.api.data_processor import DataProcessor

def calculate_point_at_angle(center_lat, center_lon, angle_degrees, distance=0.001):
    """
    Calculate a point at a given angle and distance from a center point.
    
    Args:
        center_lat (float): Center latitude
        center_lon (float): Center longitude
        angle_degrees (float): Angle in degrees (0=North, 90=East, 180=South, 270=West)
        distance (float): Distance in degrees (approximately 0.001 = 100m)
        
    Returns:
        tuple: (latitude, longitude)
    """
    # Convert angle to radians (0 degrees = North)
    angle_rad = math.radians(90 - angle_degrees)
    
    # Calculate offsets
    lat_offset = distance * math.cos(angle_rad)
    lon_offset = distance * math.sin(angle_rad)
    
    return center_lat + lat_offset, center_lon + lon_offset

def calculate_missing_metrics(approach_data):
    """Calculate missing metrics for an approach if they're not already present."""
    if approach_data is None:
        return {}
        
    # Calculate speed ratio if missing
    if "speed_ratio" not in approach_data and "current_speed" in approach_data and "free_flow_speed" in approach_data:
        current = approach_data["current_speed"]
        free_flow = approach_data["free_flow_speed"]
        if current is not None and free_flow is not None and free_flow > 0:
            approach_data["speed_ratio"] = round(current / free_flow, 2)
    
    # Determine congestion level if missing
    if "congestion_level" not in approach_data and "speed_ratio" in approach_data:
        ratio = approach_data["speed_ratio"]
        if ratio >= 0.9:
            approach_data["congestion_level"] = "Free Flow"
        elif ratio >= 0.75:
            approach_data["congestion_level"] = "Light"
        elif ratio >= 0.5:
            approach_data["congestion_level"] = "Moderate"
        elif ratio >= 0.25:
            approach_data["congestion_level"] = "Heavy"
        else:
            approach_data["congestion_level"] = "Severe"
    
    # Calculate traffic score if missing
    if "traffic_score" not in approach_data and "speed_ratio" in approach_data:
        ratio = approach_data["speed_ratio"]
        delay_ratio = approach_data.get("delay", 0) / 100  # Normalize delay
        score = (ratio * 0.7 + (1 - delay_ratio) * 0.3) * 100
        approach_data["traffic_score"] = round(min(100, max(0, score)))
    
    return approach_data

def get_marker_color(approach_data):
    """Determine marker color based on traffic conditions."""
    # Default to blue if no data
    if not approach_data:
        return "blue"
        
    # If we have a congestion level, use that
    congestion = approach_data.get("congestion_level")
    if congestion:
        if congestion == "Free Flow":
            return "green"
        elif congestion == "Light":
            return "lightgreen"
        elif congestion == "Moderate":
            return "orange"
        elif congestion == "Heavy":
            return "red"
        elif congestion == "Severe":
            return "darkred"
    
    # If we have speed ratio, use that
    ratio = approach_data.get("speed_ratio")
    if ratio:
        if ratio >= 0.9:
            return "green"
        elif ratio >= 0.75:
            return "lightgreen"
        elif ratio >= 0.5:
            return "orange"
        elif ratio >= 0.25:
            return "red"
        else:
            return "darkred"
    
    # If we have delay, use that
    delay = approach_data.get("delay")
    if delay:
        if delay < 10:
            return "green"
        elif delay < 30:
            return "lightgreen"
        elif delay < 60:
            return "orange"
        elif delay < 120:
            return "red"
        else:
            return "darkred"
            
    return "blue"  # Default

def fetch_traffic_data(intersection_lat, intersection_lon, use_cached=False):
    """Fetch traffic data from API or load from cached file."""
    if use_cached:
        # Load from cached file
        json_file_path = os.path.join(project_root, "traffic_summary.json")
        try:
            with open(json_file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cached data: {e}")
            return None
    
    # Fetch from API
    try:
        # Load API key
        load_dotenv()
        api_key = os.getenv("TOMTOM_API_KEY")
        if not api_key:
            print("No API key found. Please set TOMTOM_API_KEY in your .env file.")
            return None
            
        # Create client and fetch data
        client = TomTomClient(api_key=api_key)
        traffic_data = client.get_traffic_summary(intersection_lat, intersection_lon)
        
        # Save the data for future reference
        with open("traffic_summary.json", 'w') as f:
            json.dump(traffic_data, f, indent=2)
            
        return traffic_data
    except Exception as e:
        print(f"Error fetching traffic data: {e}")
        traceback.print_exc()
        return None

def create_intersection_map(traffic_data, output_file="intersection_traffic_map.html", show_all_approaches=True):
    """Create a folium map visualization for an intersection."""
    if not traffic_data or "intersection" not in traffic_data:
        print("Invalid traffic data provided")
        return None

    # Extract intersection coordinates
    intersection = traffic_data["intersection"]
    intersection_lat = intersection["latitude"]
    intersection_lon = intersection["longitude"]

    # Create a map centered on the intersection
    m = folium.Map(location=[intersection_lat, intersection_lon], zoom_start=17)

    # Add a center marker for the intersection
    folium.Marker(
        location=[intersection_lat, intersection_lon],
        popup="Intersection Center",
        icon=folium.Icon(color="blue", icon="traffic-light", prefix="fa")
    ).add_to(m)

    # Define all eight approaches for this intersection with precise angles and distances
    # These angles and distances are carefully tuned to place markers on actual road lanes
    approaches = {
        # Major arterial 1: Cobb Parkway (N-S diagonal)
        "NorthboundCobb": {
            "name": "Northbound Cobb Pkwy",
            "angle": 345,  # Aligned with actual road direction
            "distance": 0.00125,
            "icon": "car",
            "main": True
        },
        "SouthboundCobb": {
            "name": "Southbound Cobb Pkwy",
            "angle": 165,  # Aligned with actual road direction
            "distance": 0.00125,
            "icon": "car",
            "main": True
        },
        
        # Major arterial 2: North Marietta Parkway (E-W diagonal)
        "EastboundMarietta": {
            "name": "Eastbound N Marietta Pkwy",
            "angle": 75,  # Aligned with actual road direction
            "distance": 0.00125,
            "icon": "car",
            "main": True
        },
        "WestboundMarietta": {
            "name": "Westbound N Marietta Pkwy",
            "angle": 255,  # Aligned with actual road direction 
            "distance": 0.00125,
            "icon": "car",
            "main": True
        },
        
        # Diagonal approaches (secondary roads and ramps)
        "NortheastRamp": {
            "name": "Northeast Approach",
            "angle": 35,  # Adjusted to match ramp
            "distance": 0.0012,
            "icon": "car",
            "main": False
        },
        "NorthwestRamp": {
            "name": "Northwest Approach",
            "angle": 305,  # Adjusted to match ramp
            "distance": 0.0012,
            "icon": "car",
            "main": False
        },
        "SoutheastRamp": {
            "name": "Southeast Approach",
            "angle": 125,  # Adjusted to match ramp
            "distance": 0.0012,
            "icon": "car",
            "main": False
        },
        "SouthwestRamp": {
            "name": "Southwest Approach",
            "angle": 215,  # Adjusted to match ramp
            "distance": 0.0012,
            "icon": "car",
            "main": False
        }
    }

    # Process and add approach markers
    for direction, config in approaches.items():
        # Skip secondary approaches if not showing all
        if not config.get("main", True) and not show_all_approaches:
            continue
            
        # Calculate approach coordinates using angle-based approach for better accuracy
        approach_lat, approach_lon = calculate_point_at_angle(
            intersection_lat, 
            intersection_lon, 
            config["angle"], 
            config["distance"]
        )
        
        # Get traffic data for this approach
        # Map the code direction key to whatever direction keys are in your data
        direction_key = direction  # You might need to map this to your data structure
        approach_data = traffic_data.get("approaches", {}).get(direction_key, {})
        
        # If no data for this specific direction, try to find a generic direction match
        if not approach_data:
            # Try to match with more generic directions (N, S, E, W, etc)
            generic_directions = {
                "NorthboundCobb": ["North", "Northbound"],
                "SouthboundCobb": ["South", "Southbound"],
                "EastboundMarietta": ["East", "Eastbound"],
                "WestboundMarietta": ["West", "Westbound"],
                "NortheastRamp": ["Northeast"],
                "NorthwestRamp": ["Northwest"],
                "SoutheastRamp": ["Southeast"],
                "SouthwestRamp": ["Southwest"]
            }
            
            for generic in generic_directions.get(direction, []):
                if generic in traffic_data.get("approaches", {}):
                    approach_data = traffic_data["approaches"][generic]
                    break
        
        # Skip completely if no data and it's a secondary approach (unless showing all)
        if not approach_data and not config.get("main", True) and not show_all_approaches:
            continue
        
        # Calculate any missing metrics
        approach_data = calculate_missing_metrics(approach_data)
        
        # Determine marker color based on traffic conditions
        color = get_marker_color(approach_data)
        if not approach_data:
            color = "gray"  # Use gray for approaches with no data
        
        # Create popup content
        if approach_data:
            popup_content = f"""
            <b>{config["name"]}</b><br>
            Current Speed: {approach_data.get('current_speed', 'N/A')} km/h<br>
            Free Flow Speed: {approach_data.get('free_flow_speed', 'N/A')} km/h<br>
            Speed Ratio: {approach_data.get('speed_ratio', 'N/A')}<br>
            Delay: {approach_data.get('delay', 'N/A')} seconds<br>
            Congestion Level: {approach_data.get('congestion_level', 'N/A')}<br>
            Traffic Score: {approach_data.get('traffic_score', 'N/A')}
            """
        else:
            popup_content = f"<b>{config['name']}</b><br>No traffic data available"
        
        # Add marker to map with appropriate icon and rotation
        folium.Marker(
            location=[approach_lat, approach_lon],
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=config["name"],
            icon=folium.Icon(color=color, icon=config["icon"], prefix="fa")
        ).add_to(m)

        # Draw a line from center to approach
        folium.PolyLine(
            locations=[[intersection_lat, intersection_lon], [approach_lat, approach_lon]],
            color=color,
            weight=2,
            opacity=0.7,
            tooltip=config["name"],
            dash_array="5, 5" if not config.get("main", True) else None  # Dashed line for secondary approaches
        ).add_to(m)

    # Add a circle to represent the intersection zone (perfectly centered)
    folium.Circle(
        location=[intersection_lat, intersection_lon],
        radius=80,  # meters - slightly smaller for better visualization
        color="gray",
        fill=True,
        fill_opacity=0.1,
        tooltip="Intersection Zone"
    ).add_to(m)

    # Add a custom legend to explain the visualization
    legend_html = """
    <div style="position: fixed; 
        bottom: 50px; right: 50px; width: 180px; height: 200px; 
        border:2px solid grey; z-index:9999; background-color:white;
        padding: 10px; font-size: 14px; border-radius: 5px;">
        <p><b>Traffic Visualization</b></p>
        <p><i class="fa fa-car" style="color:green;"></i> Free Flow</p>
        <p><i class="fa fa-car" style="color:orange;"></i> Moderate Congestion</p>
        <p><i class="fa fa-car" style="color:red;"></i> Heavy Congestion</p>
        <p><i class="fa fa-car" style="color:gray;"></i> No Data</p>
        <p><i class="fa fa-traffic-light" style="color:blue;"></i> Intersection Center</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save the map
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "." in output_file:
        base, ext = output_file.rsplit(".", 1)
        output_file = f"{base}_{timestamp}.{ext}"
    else:
        output_file = f"{output_file}_{timestamp}.html"
            
    m.save(output_file)
    print(f"Map saved to {output_file}")
    return output_file

def main():
    """Main entry point for the traffic visualization script."""
    import argparse
    parser = argparse.ArgumentParser(description="Traffic Visualization")
    parser.add_argument("--lat", type=float, default=33.960192828395996, 
                       help="Intersection latitude")
    parser.add_argument("--lon", type=float, default=-84.52790520126695, 
                       help="Intersection longitude")
    parser.add_argument("--cached", action="store_true", 
                       help="Use cached data instead of fetching from API")
    parser.add_argument("--output", type=str, default="intersection_traffic_map.html",
                       help="Output HTML file name")
    parser.add_argument("--all-approaches", action="store_true", default=True,
                       help="Show all eight approaches (including diagonal roads)")
    args = parser.parse_args()
    
    # Fetch traffic data
    print(f"Fetching traffic data for intersection at {args.lat}, {args.lon}...")
    traffic_data = fetch_traffic_data(args.lat, args.lon, args.cached)
    
    if not traffic_data:
        print("Failed to get traffic data. Exiting.")
        return
    
    # Create visualization
    print("Creating traffic visualization...")
    output_file = create_intersection_map(
        traffic_data, 
        args.output, 
        show_all_approaches=args.all_approaches
    )
    
    print(f"Visualization complete: {output_file}")

if __name__ == "__main__":
    main()