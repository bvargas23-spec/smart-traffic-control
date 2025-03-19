"""
Test Data Processor Module

This script tests the DataProcessor class to ensure it correctly
processes and analyzes traffic data.
"""

import os
import sys
import json
import pytest
from datetime import datetime
import pandas as pd
import numpy as np

# Add the src directory to the Python path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.data_processor import DataProcessor

# Create a temporary data directory for tests
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# Create a fixture for the data processor that will be used in multiple tests
@pytest.fixture
def processor():
    return DataProcessor(data_dir=TEST_DATA_DIR)

# Sample data fixtures
@pytest.fixture
def sample_flow_data():
    return {
        "flowSegmentData": {
            "frc": "FRC0",
            "currentSpeed": 45,
            "freeFlowSpeed": 60,
            "currentTravelTime": 120,
            "freeFlowTravelTime": 90,
            "confidence": 0.95,
            "roadClosure": False
        }
    }

@pytest.fixture
def sample_intersection_data():
    return {
        "intersection": {
            "latitude": 33.960192,
            "longitude": -84.527905
        },
        "approaches": {
            "North": {
                "current_speed": 48,
                "free_flow_speed": 60,
                "current_travel_time": 110,
                "free_flow_travel_time": 90,
                "confidence": 0.9
            },
            "South": {
                "current_speed": 25,
                "free_flow_speed": 55,
                "current_travel_time": 180,
                "free_flow_travel_time": 100,
                "confidence": 0.8
            },
            "East": {
                "current_speed": 32,
                "free_flow_speed": 50,
                "current_travel_time": 150,
                "free_flow_travel_time": 100,
                "confidence": 0.95
            },
            "West": {
                "current_speed": 30,
                "free_flow_speed": 50,
                "current_travel_time": 160,
                "free_flow_travel_time": 100,
                "confidence": 0.9
            }
        }
    }

@pytest.fixture
def sample_incidents_data():
    return {
        "incidents": [
            {
                "type": "ACCIDENT",
                "properties": {
                    "events": [
                        {
                            "description": "Accident on Main St",
                            "code": 401
                        }
                    ],
                    "delay": 300,
                    "iconCategory": "ACCIDENT"
                }
            },
            {
                "type": "CONSTRUCTION",
                "properties": {
                    "events": [
                        {
                            "description": "Construction on Oak Ave",
                            "code": 501
                        }
                    ],
                    "delay": 120,
                    "iconCategory": "CONSTRUCTION"
                }
            }
        ]
    }

def test_process_traffic_flow(processor, sample_flow_data):
    """Test processing of traffic flow data."""
    processed_data = processor.process_traffic_flow(sample_flow_data)
    
    # Assert that key metrics are calculated correctly
    assert processed_data["current_speed"] == 45
    assert processed_data["free_flow_speed"] == 60
    assert processed_data["speed_ratio"] == 0.75  # 45/60
    assert processed_data["delay"] == 30  # 120-90
    assert processed_data["congestion_level"] == "Light"  # Based on speed ratio of 0.75
    
    # Check that the traffic score is calculated
    assert "traffic_score" in processed_data
    assert 50 <= processed_data["traffic_score"] <= 80  # Expected range for this data
    
    # Check that timestamp is added
    assert "timestamp" in processed_data

def test_process_intersection_data(processor, sample_intersection_data):
    """Test processing of intersection data with multiple approaches."""
    processed_data = processor.process_intersection_data(sample_intersection_data)
    
    # Check for intersection metrics
    assert "intersection" in processed_data
    assert "approaches" in processed_data
    
    # Check that each approach is processed
    approaches = processed_data["approaches"]
    assert "North" in approaches
    assert "South" in approaches
    assert "East" in approaches
    assert "West" in approaches
    
    # Check derived metrics for one approach
    north = approaches["North"]
    assert "speed_ratio" in north
    assert "congestion_level" in north
    assert "traffic_score" in north
    
    # Check that intersection-level metrics are calculated
    intersection = processed_data["intersection"]
    assert "average_score" in intersection
    assert "weighted_score" in intersection
    assert "most_congested_approach" in intersection
    assert "overall_status" in intersection

def test_process_incidents(processor, sample_incidents_data):
    """Test processing of traffic incidents data."""
    processed_incidents = processor.process_incidents(sample_incidents_data)
    
    # Check that we have the correct number of incidents
    assert len(processed_incidents) == 2
    
    # Check that the first incident is processed correctly
    incident = processed_incidents[0]
    assert incident["type"] == "ACCIDENT"
    assert incident["description"] == "Accident on Main St"
    assert incident["delay"] == 300
    assert "severity" in incident
    
    # Check that severity is calculated
    assert incident["severity"] in ["Low", "Minor", "Moderate", "Major", "Severe"]

def test_save_and_load_processed_data(processor, sample_intersection_data):
    """Test saving and loading processed data."""
    # Process some data
    processed_data = processor.process_intersection_data(sample_intersection_data)
    
    # Save it to a file
    filename = "test_save_load.json"
    file_path = processor.save_processed_data(processed_data, filename)
    
    # Check that the file exists
    assert os.path.exists(file_path)
    
    # Load the data back
    loaded_data = processor.load_processed_data(file_path)
    
    # Verify it's the same data (comparison needs to handle dynamically generated timestamps)
    assert loaded_data["intersection"]["latitude"] == processed_data["intersection"]["latitude"]
    assert loaded_data["intersection"]["longitude"] == processed_data["intersection"]["longitude"]
    assert loaded_data["approaches"]["North"]["current_speed"] == processed_data["approaches"]["North"]["current_speed"]
    
    # Clean up the test file
    os.remove(file_path)

def test_append_to_time_series(processor, sample_intersection_data):
    """Test appending data to a time series CSV file."""
    # Process some data
    processed_data = processor.process_intersection_data(sample_intersection_data)
    
    # Append to time series
    test_csv = "test_time_series.csv"
    file_path = processor.append_to_time_series(processed_data, test_csv)
    
    # Check that the file exists
    assert os.path.exists(file_path)
    
    # Read the CSV and check it has the expected structure
    df = pd.read_csv(file_path)
    assert "timestamp" in df.columns
    assert "latitude" in df.columns
    assert "longitude" in df.columns
    assert "weighted_score" in df.columns
    
    # Check that the data matches
    assert df.iloc[0]["latitude"] == sample_intersection_data["intersection"]["latitude"]
    
    # Append again to ensure we can add multiple rows
    processor.append_to_time_series(processed_data, test_csv)
    df = pd.read_csv(file_path)
    assert len(df) == 2  # Should now have 2 rows
    
    # Clean up the test file
    os.remove(file_path)

def test_calculate_optimal_cycle_times(processor, sample_intersection_data):
    """Test calculation of optimal traffic signal cycle times."""
    # Process some data
    processed_data = processor.process_intersection_data(sample_intersection_data)
    
    # Calculate optimal cycle times
    timings = processor.calculate_optimal_cycle_times(processed_data)
    
    # Check that the timing data has the expected structure
    assert "cycle_length" in timings
    assert "phase_times" in timings
    assert "North-South" in timings["phase_times"]
    assert "East-West" in timings["phase_times"]
    
    # Check that the cycle length is within reasonable bounds
    assert 30 <= timings["cycle_length"] <= 120
    
    # Check that the green times add up correctly
    total_green = sum(phase["green_time"] for phase in timings["phase_times"].values())
    total_lost = timings["total_lost_time"]
    assert abs(total_green + total_lost - timings["cycle_length"]) <= 1  # Allow for rounding

def test_congestion_classification(processor):
    """Test the congestion classification logic."""
    # Test different speed ratios
    assert processor._classify_congestion(0.95) == "Free Flow"
    assert processor._classify_congestion(0.8) == "Light"
    assert processor._classify_congestion(0.6) == "Moderate"
    assert processor._classify_congestion(0.3) == "Heavy"
    assert processor._classify_congestion(0.1) == "Severe"

def test_traffic_score_calculation(processor):
    """Test the traffic score calculation logic."""
    # Test different combinations of inputs
    score1 = processor._calculate_traffic_score(1.0, 0.0, 1.0)  # Perfect conditions
    score2 = processor._calculate_traffic_score(0.5, 0.5, 1.0)  # Moderate conditions
    score3 = processor._calculate_traffic_score(0.2, 0.8, 1.0)  # Poor conditions
    
    # Check that scores are in the expected ranges
    assert score1 > 90  # Near-perfect score
    assert 40 <= score2 <= 60  # Mid-range score
    assert score3 < 30  # Low score
    
    # Check that confidence affects the score
    assert processor._calculate_traffic_score(0.8, 0.2, 0.5) < processor._calculate_traffic_score(0.8, 0.2, 1.0)

def test_incident_severity_calculation(processor):
    """Test the incident severity calculation logic."""
    # Test different delay values
    assert processor._calculate_incident_severity(1000, "ACCIDENT") == "Severe"
    assert processor._calculate_incident_severity(700, "CONSTRUCTION") == "Major"
    assert processor._calculate_incident_severity(400, "CONGESTION") == "Moderate"
    assert processor._calculate_incident_severity(100, "UNKNOWN") == "Minor"
    assert processor._calculate_incident_severity(30, "OTHER") == "Low"
    
    # Test category overrides
    assert processor._calculate_incident_severity(30, "ACCIDENT") == "Moderate"  # Upgraded from Low
    
def test_generate_time_series_analysis(processor):
    """Test time series analysis with a synthetic dataset."""
    # Create a synthetic time series CSV for testing
    test_csv = "test_analysis.csv"
    file_path = os.path.join(TEST_DATA_DIR, test_csv)
    
    # Create synthetic data
    now = datetime.now()
    data = []
    for day in range(7):  # One week of data
        for hour in range(24):  # 24 hours per day
            timestamp = now.replace(day=now.day-7+day, hour=hour, minute=0, second=0)
            
            # Simulate traffic patterns (worse during rush hours)
            rush_hour_factor = 1.0
            if hour in [7, 8, 9, 16, 17, 18]:  # Rush hours
                rush_hour_factor = 0.6
            elif hour >= 22 or hour <= 5:  # Late night/early morning
                rush_hour_factor = 1.3
                
            # Weekend factor (better traffic on weekends)
            weekend_factor = 1.2 if day >= 5 else 1.0  # Days 5-6 are weekend
            
            # Calculate score (higher is better)
            score = 70 * rush_hour_factor * weekend_factor
            
            # Add some randomness
            score = max(0, min(100, score + np.random.normal(0, 5)))
            
            # Create a row
            row = {
                "timestamp": timestamp.isoformat(),
                "latitude": 33.96,
                "longitude": -84.53,
                "weighted_score": score,
                "average_score": score * 0.95,  # Slightly lower than weighted
                "max_delay": 120 if score < 50 else 60 if score < 80 else 20,
                "total_delay": 300 if score < 50 else 150 if score < 80 else 50,
                "overall_status": "Congested" if score < 50 else "Moderate" if score < 80 else "Good",
                "North_speed": 30 * (score/100),
                "North_congestion": "Severe" if score < 40 else "Heavy" if score < 60 else "Moderate" if score < 75 else "Light" if score < 90 else "Free Flow",
                "South_speed": 25 * (score/100),
                "South_congestion": "Severe" if score < 30 else "Heavy" if score < 50 else "Moderate" if score < 70 else "Light" if score < 85 else "Free Flow"
            }
            data.append(row)
    
    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    
    # Run the analysis
    analysis = processor.analyze_time_series(test_csv)
    
    # Check the analysis results
    assert "data_points" in analysis
    assert analysis["data_points"] == len(data)
    
    assert "hourly_patterns" in analysis
    assert "peak_congestion_hour" in analysis["hourly_patterns"]
    
    assert "daily_patterns" in analysis
    assert "worst_day" in analysis["daily_patterns"]
    
    # Clean up
    os.remove(file_path)

# Run the tests if executed directly
if __name__ == "__main__":
    # Create a data processor
    processor = DataProcessor(data_dir=TEST_DATA_DIR)
    
    # Sample data for testing
    sample_flow_data = {
        "flowSegmentData": {
            "frc": "FRC0",
            "currentSpeed": 45,
            "freeFlowSpeed": 60,
            "currentTravelTime": 120,
            "freeFlowTravelTime": 90,
            "confidence": 0.95,
            "roadClosure": False
        }
    }
    
    # Process flow data
    print("\n=== Testing Traffic Flow Processing ===")
    processed_flow = processor.process_traffic_flow(sample_flow_data)
    print(f"Speed Ratio: {processed_flow['speed_ratio']}")
    print(f"Congestion Level: {processed_flow['congestion_level']}")
    print(f"Traffic Score: {processed_flow['traffic_score']}")
    
    # Test intersection processing
    print("\n=== Testing Intersection Processing ===")
    sample_intersection_data = {
        "intersection": {
            "latitude": 33.960192,
            "longitude": -84.527905
        },
        "approaches": {
            "North": {
                "current_speed": 48,
                "free_flow_speed": 60,
                "current_travel_time": 110,
                "free_flow_travel_time": 90,
                "delay": 20,
                "confidence": 0.9
            },
            "South": {
                "current_speed": 25,
                "free_flow_speed": 55,
                "current_travel_time": 180,
                "free_flow_travel_time": 100,
                "delay": 80,
                "confidence": 0.8
            },
            "East": {
                "current_speed": 32,
                "free_flow_speed": 50,
                "current_travel_time": 150,
                "free_flow_travel_time": 100,
                "delay": 50,
                "confidence": 0.95
            },
            "West": {
                "current_speed": 30,
                "free_flow_speed": 50,
                "current_travel_time": 160,
                "free_flow_travel_time": 100,
                "delay": 60,
                "confidence": 0.9
            }
        }
    }
    
    processed_intersection = processor.process_intersection_data(sample_intersection_data)
    print("Intersection Score:", processed_intersection["intersection"].get("weighted_score"))
    print("Most Congested Approach:", processed_intersection["intersection"].get("most_congested_approach"))
    
    # Test timing calculations
    print("\n=== Testing Signal Timing Calculations ===")
    timings = processor.calculate_optimal_cycle_times(processed_intersection)
    print(f"Cycle Length: {timings['cycle_length']} seconds")
    for phase, times in timings["phase_times"].items():
        print(f"{phase}: {times['green_time']} seconds green")
    
    print("\nAll tests completed successfully!")