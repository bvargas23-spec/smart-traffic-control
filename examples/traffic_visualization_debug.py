# traffic_visualization.py - Debug Version
import os
import sys
import json
import traceback
from datetime import datetime

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Script starting at: {datetime.now()}")
print(f"Current working directory: {os.getcwd()}")
print(f"Project root directory: {project_root}")
print(f"Python path: {sys.path}")

# Check for write permissions in the current directory
try:
    test_file = "write_test.txt"
    with open(test_file, 'w') as f:
        f.write("Test write permissions")
    os.remove(test_file)
    print("Successfully verified write permissions in current directory")
except Exception as e:
    print(f"Warning: Cannot write to current directory: {e}")

# Now verify that src can be found
src_dir = os.path.join(project_root, 'src')
if not os.path.exists(src_dir):
    print(f"Error: The 'src' directory does not exist at {src_dir}")
    # List directories in project root to help diagnose
    print("Contents of project root:")
    for item in os.listdir(project_root):
        print(f"  {item}")
    sys.exit(1)

# Try to import required packages
required_packages = ['folium', 'pandas', 'python-dotenv']
missing_packages = []

try:
    import folium
    print("Successfully imported folium")
except ImportError:
    missing_packages.append('folium')

try:
    import pandas as pd
    print("Successfully imported pandas")
except ImportError:
    missing_packages.append('pandas')

try:
    from dotenv import load_dotenv
    print("Successfully imported python-dotenv")
except ImportError:
    missing_packages.append('python-dotenv')

# Install missing packages if any
if missing_packages:
    print(f"Need to install the following packages: {', '.join(missing_packages)}")
    for package in missing_packages:
        print(f"Installing {package}...")
        os.system(f"pip install {package}")
    
    # Try imports again
    try:
        import folium
        import pandas as pd
        from dotenv import load_dotenv
        print("Successfully imported all packages after installation")
    except ImportError as e:
        print(f"Error importing packages even after installation: {e}")
        sys.exit(1)

# Check for json file
json_file_path = os.path.join(project_root, "fixed_traffic_summary.json")
print(f"Checking for JSON file at: {json_file_path}")
if os.path.exists(json_file_path):
    print(f"Found JSON file: {json_file_path}")
else:
    print(f"Warning: JSON file not found at {json_file_path}")
    # Try to find it elsewhere
    for root, dirs, files in os.walk(project_root):
        if "fixed_traffic_summary.json" in files:
            json_file_path = os.path.join(root, "fixed_traffic_summary.json")
            print(f"Found JSON file at: {json_file_path}")
            break

# Now try to import the project modules
try:
    from src.api.tomtom_client import TomTomClient
    from src.api.data_processor import DataProcessor
    print("Successfully imported TomTomClient and DataProcessor")
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("\nDebugging information:")
    print("\nContents of src directory:")
    if os.path.exists(src_dir):
        for root, dirs, files in os.walk(src_dir):
            if os.path.basename(root) == 'api':
                print(f"  API directory contents: {root}")
                for file in files:
                    print(f"    File: {file}")
    sys.exit(1)

def create_intersection_visualization(intersection_lat, intersection_lon, 
                                    approach_data, output_file="intersection_visualization.html"):
    """
    Create an interactive visualization of an intersection with traffic data.
    """
    print(f"Creating visualization for intersection at ({intersection_lat}, {intersection_lon})")
    print(f"Output file will be: {os.path.abspath(output_file)}")
    
    # Create a map centered on the intersection
    m = folium.Map(location=[intersection_lat, intersection_lon], zoom_start=17)
    
    # Add a marker for the intersection center
    folium.Marker(
        location=[intersection_lat, intersection_lon],
        popup="Intersection Center",
        icon=folium.Icon(icon="crosshairs", prefix="fa", color="blue")
    ).add_to(m)
    
    # Calculate approach positions (approximate positions for visualization)
    approach_positions = {
        "North": (intersection_lat + 0.0005, intersection_lon),
        "South": (intersection_lat - 0.0005, intersection_lon),
        "East": (intersection_lat, intersection_lon + 0.0005),
        "West": (intersection_lat, intersection_lon - 0.0005),
        "Northeast": (intersection_lat + 0.00035, intersection_lon + 0.00035),
        "Northwest": (intersection_lat + 0.00035, intersection_lon - 0.00035),
        "Southeast": (intersection_lat - 0.00035, intersection_lon + 0.00035),
        "Southwest": (intersection_lat - 0.00035, intersection_lon - 0.00035)
    }
    
    # Add markers for each approach with traffic data popups
    print(f"Adding markers for {len(approach_data)} approaches")
    for direction, data in approach_data.items():
        if "error" in data:
            print(f"Skipping {direction} approach due to error: {data.get('error')}")
            continue
            
        # Get position for this approach
        pos = approach_positions.get(direction)
        if not pos:
            print(f"No position defined for {direction} approach, skipping")
            continue
            
        print(f"Adding marker for {direction} approach")
        # Determine color based on congestion level
        color = get_marker_color(data)
        
        # Create popup content with traffic information
        popup_content = f"""
        <h4>{direction} Approach</h4>
        <table>
            <tr><td>Current Speed:</td><td>{data.get('current_speed', 'N/A')} km/h</td></tr>
            <tr><td>Free Flow Speed:</td><td>{data.get('free_flow_speed', 'N/A')} km/h</td></tr>
            <tr><td>Speed Ratio:</td><td>{data.get('speed_ratio', 'N/A')}</td></tr>
            <tr><td>Delay:</td><td>{data.get('delay', 'N/A')} seconds</td></tr>
            <tr><td>Congestion Level:</td><td>{data.get('congestion_level', 'N/A')}</td></tr>
            <tr><td>Traffic Score:</td><td>{data.get('traffic_score', 'N/A')}</td></tr>
        </table>
        """
        
        # Add marker to map
        folium.Marker(
            location=pos,
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=f"{direction}: {data.get('congestion_level', 'N/A')}",
            icon=folium.Icon(icon="road", prefix="fa", color=color)
        ).add_to(m)
        
        # Draw a line from the center to the approach
        folium.PolyLine(
            locations=[[intersection_lat, intersection_lon], pos],
            color=color,
            weight=3,
            opacity=0.7,
            tooltip=f"{direction} approach"
        ).add_to(m)
    
    # Add a circle to represent the intersection area
    folium.Circle(
        location=[intersection_lat, intersection_lon],
        radius=50,  # meters
        color="gray",
        fill=True,
        fill_opacity=0.2
    ).add_to(m)
    
    # Add a color-coded legend for traffic conditions
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px">
    <h4>Traffic Conditions</h4>
    <div><i style="background: green; width: 15px; height: 15px; display: inline-block"></i> Free Flow</div>
    <div><i style="background: lightgreen; width: 15px; height: 15px; display: inline-block"></i> Light</div>
    <div><i style="background: orange; width: 15px; height: 15px; display: inline-block"></i> Moderate</div>
    <div><i style="background: red; width: 15px; height: 15px; display: inline-block"></i> Heavy</div>
    <div><i style="background: darkred; width: 15px; height: 15px; display: inline-block"></i> Severe</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save the map to HTML
    try:
        print(f"Saving visualization to {output_file}")
        m.save(output_file)
        print(f"Visualization saved successfully to {os.path.abspath(output_file)}")
        return os.path.abspath(output_file)
    except Exception as e:
        print(f"Error saving visualization: {e}")
        traceback.print_exc()
        raise

def get_marker_color(traffic_data):
    """
    Determine marker color based on traffic conditions.
    """
    # Get congestion level or calculate from speed ratio
    congestion_level = traffic_data.get('congestion_level')
    
    if not congestion_level and 'speed_ratio' in traffic_data:
        speed_ratio = traffic_data.get('speed_ratio', 0)
        if speed_ratio >= 0.9:
            congestion_level = "Free Flow"
        elif speed_ratio >= 0.75:
            congestion_level = "Light"
        elif speed_ratio >= 0.5:
            congestion_level = "Moderate"
        elif speed_ratio >= 0.25:
            congestion_level = "Heavy"
        else:
            congestion_level = "Severe"
    
    # Map congestion level to color
    color_map = {
        "Free Flow": "green",
        "Light": "lightgreen",
        "Moderate": "orange",
        "Heavy": "red",
        "Severe": "darkred"
    }
    
    return color_map.get(congestion_level, "blue")

def visualize_from_file(json_file):
    """
    Create a visualization from a previously saved JSON file.
    """
    print(f"Visualizing from file: {json_file}")
    
    try:
        # Load the data from the file
        print(f"Reading JSON data from {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if "intersection" not in data or "approaches" not in data:
            print(f"Error: Invalid data format in JSON file. Missing 'intersection' or 'approaches' keys.")
            print(f"Available keys: {list(data.keys())}")
            raise ValueError("Invalid data format in JSON file")
        
        # Extract intersection coordinates
        intersection_lat = data["intersection"]["latitude"]
        intersection_lon = data["intersection"]["longitude"]
        print(f"Intersection coordinates: ({intersection_lat}, {intersection_lon})")
        
        # Create visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(os.getcwd(), f"intersection_visualization_{timestamp}.html")
        
        return create_intersection_visualization(
            intersection_lat,
            intersection_lon,
            data["approaches"],
            output_file
        )
        
    except Exception as e:
        print(f"Error creating visualization from file: {e}")
        traceback.print_exc()
        raise

# Example usage
if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    print("Starting traffic visualization...")
    
    # Option: Visualize from the fixed_traffic_summary.json file
    try:
        print("\nVisualizing from fixed_traffic_summary.json file")
        if os.path.exists(json_file_path):
            print(f"Using JSON file at: {json_file_path}")
            html_file = visualize_from_file(json_file_path)
            print(f"\nVisualization complete!")
            print(f"Open this file in your web browser to view the visualization:")
            print(f"{html_file}")
        else:
            print(f"Error: Could not find {json_file_path}")
            # Try with a relative path instead
            alt_path = "fixed_traffic_summary.json"
            if os.path.exists(alt_path):
                print(f"Found JSON file at: {os.path.abspath(alt_path)}")
                html_file = visualize_from_file(alt_path)
                print(f"\nVisualization complete!")
                print(f"Open this file in your web browser to view the visualization:")
                print(f"{html_file}")
            else:
                print("Error: Could not find fixed_traffic_summary.json in any location")
    except Exception as e:
        print(f"Error in visualization process: {e}")
        traceback.print_exc()