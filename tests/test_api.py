"""
Test TomTom API Client

This script tests the TomTomClient class to ensure it correctly
fetches traffic data from TomTom APIs.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the src directory to the Python path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.tomtom_client import TomTomClient

def test_traffic_flow(client):
    # Use the passed client parameter
    """Test getting traffic flow data for a specific point."""
    print("\n=== Testing Traffic Flow API ===")
    
    # North Marietta Pkwy Ne & Cobb Pkwy N intersection
    lat = 33.960192828395996
    lon = -84.52790520126695
    
    try:
        # Get traffic flow data
        flow_data = client.get_traffic_flow(lat, lon)
        
        # Check if we got valid data
        if 'flowSegmentData' in flow_data:
            print("✅ Successfully retrieved traffic flow data")
            
            # Print some key metrics
            segment = flow_data['flowSegmentData']
            print(f"  Current Speed: {segment.get('currentSpeed')} kph")
            print(f"  Free Flow Speed: {segment.get('freeFlowSpeed')} kph")
            print(f"  Current Travel Time: {segment.get('currentTravelTime')} seconds")
            
            return True
        else:
            print("❌ Received response but no flow data found")
            return False
            
    except Exception as e:
        print(f"❌ Error retrieving traffic flow data: {e}")
        return False

def test_traffic_incidents(client):
    """Test getting traffic incidents within a bounding box."""
    print("\n=== Testing Traffic Incidents API ===")
    
    # Define a bounding box around the intersection (approximately 500m in each direction)
    lat = 33.960192828395996
    lon = -84.52790520126695
    
    bbox = (
        lon - 0.005,  # min_lon
        lat - 0.005,  # min_lat
        lon + 0.005,  # max_lon
        lat + 0.005   # max_lat
    )
    
    try:
        # Get incidents data
        incidents_data = client.get_traffic_incidents_in_bbox(bbox)
        
        # Check if we got valid data
        if 'incidents' in incidents_data:
            incidents = incidents_data['incidents']
            print(f"✅ Successfully retrieved {len(incidents)} traffic incidents")
            
            # Print some details of incidents if any
            if incidents:
                for i, incident in enumerate(incidents[:3]):  # Show max 3 incidents
                    incident_type = incident.get('type', 'Unknown')
                    properties = incident.get('properties', {})
                    delay = properties.get('delay', 'Unknown')
                    
                    print(f"  Incident {i+1}: Type={incident_type}, Delay={delay} seconds")
            else:
                print("  No incidents reported in this area currently")
                
            return True
        else:
            print("❌ Received response but no incidents data found")
            return False
            
    except Exception as e:
        print(f"❌ Error retrieving traffic incidents data: {e}")
        return False

def test_intersection_approaches(client):
    """Test getting traffic data for all approaches to an intersection."""
    print("\n=== Testing Intersection Approaches ===")
    
    # North Marietta Pkwy Ne & Cobb Pkwy N intersection
    lat = 33.960192828395996
    lon = -84.52790520126695
    
    try:
        # Get approach data
        approaches_data = client.get_intersection_approaches(lat, lon)
        
        # Check if we got valid data for all approaches
        success = True
        for direction, data in approaches_data.items():
            if "error" in data:
                print(f"❌ Error getting data for {direction} approach: {data['error']}")
                success = False
                continue
                
            if "flowSegmentData" not in data:
                print(f"❌ No flow data found for {direction} approach")
                success = False
                continue
                
            # Data looks good for this approach
            segment = data["flowSegmentData"]
            current_speed = segment.get("currentSpeed", "N/A")
            free_flow_speed = segment.get("freeFlowSpeed", "N/A")
            
            print(f"  {direction}: Current Speed={current_speed}kph, Free Flow={free_flow_speed}kph")
        
        if success:
            print("✅ Successfully retrieved data for all approaches")
        
        return success
            
    except Exception as e:
        print(f"❌ Error retrieving approach data: {e}")
        return False

def test_traffic_summary(client):
    """Test getting a comprehensive traffic summary for an intersection."""
    print("\n=== Testing Traffic Summary ===")
    
    # North Marietta Pkwy Ne & Cobb Pkwy N intersection
    lat = 33.960192828395996
    lon = -84.52790520126695
    
    try:
        # Get traffic summary
        summary = client.get_traffic_summary(lat, lon)
        
        # Check if we got valid data
        if "intersection" in summary and "approaches" in summary:
            print("✅ Successfully retrieved traffic summary")
            
            # Print some summary statistics
            approaches = summary["approaches"]
            
            # Count approaches with congestion (defined as having delay)
            congested_approaches = sum(1 for approach in approaches.values() 
                                     if isinstance(approach.get("delay"), (int, float)) 
                                     and approach["delay"] > 10)
            
            total_approaches = len(approaches)
            
            print(f"  Congested Approaches: {congested_approaches}/{total_approaches}")
            
            # Print the most congested approach
            max_delay = -1
            most_congested = None
            
            for direction, data in approaches.items():
                delay = data.get("delay", 0)
                if isinstance(delay, (int, float)) and delay > max_delay:
                    max_delay = delay
                    most_congested = direction
            
            if most_congested:
                print(f"  Most Congested Approach: {most_congested} (Delay: {max_delay}s)")
            else:
                print("  No significant congestion detected")
            
            # Save the full summary to a JSON file for inspection
            with open("traffic_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
                
            print("  Full traffic summary saved to traffic_summary.json")
            
            return True
        else:
            print("❌ Received response but summary data is incomplete")
            return False
            
    except Exception as e:
        print(f"❌ Error retrieving traffic summary: {e}")
        return False

# Main test runner
if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment variables
    API_KEY = os.getenv("TOMTOM_API_KEY")
    if not API_KEY:
        print("❌ No API key found. Please set TOMTOM_API_KEY in your .env file.")
        sys.exit(1)
    
    # Create a client
    client = TomTomClient(api_key=API_KEY)
    
    # Run the tests
    tests = [
        test_traffic_flow,
        test_traffic_incidents,
        test_intersection_approaches,
        test_traffic_summary
    ]
    
    # Keep track of results
    results = []
    
    # Run each test
    for test in tests:
        results.append(test())
    
    # Print overall results
    print("\n=== Test Results ===")
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed.")