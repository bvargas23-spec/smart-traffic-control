"""
Traffic Data Processor

This module processes raw traffic data from TomTom APIs and transforms it into
structured, analyzed information that can be used for traffic signal optimization.
It handles data cleaning, normalization, and calculation of key traffic metrics.

The DataProcessor class serves as a bridge between raw API data collection and
the decision-making algorithms for traffic control.
"""

import json
import logging
import numpy as np # type: ignore
import pandas as pd # type: ignore
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import os
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Processes raw traffic data from TomTom APIs and extracts meaningful metrics
    for traffic signal optimization.
    """
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the data processor.
        
        Args:
            data_dir (str): Directory to store processed data files
        """
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info(f"Created data directory: {data_dir}")
            
        # Default thresholds for congestion classification
        self.congestion_thresholds = {
            "free_flow": 0.9,       # >= 90% of free flow speed is considered free flowing
            "light": 0.75,          # >= 75% of free flow speed is considered light congestion
            "moderate": 0.5,        # >= 50% of free flow speed is considered moderate congestion
            "heavy": 0.25,          # >= 25% of free flow speed is considered heavy congestion
            "severe": 0             # < 25% of free flow speed is considered severe congestion
        }
        
        # Default weights for intersection health scoring
        self.approach_weights = {
            "North": 1.0,
            "South": 1.0,
            "East": 1.0,
            "West": 1.0,
            "Northeast": 0.7,
            "Northwest": 0.7,
            "Southeast": 0.7,
            "Southwest": 0.7
        }

    def process_traffic_flow(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw traffic flow data from a single point.
        
        Args:
            flow_data (dict): Raw traffic flow data from TomTom API
            
        Returns:
            dict: Processed flow data with additional metrics
        """
        if "flowSegmentData" not in flow_data:
            logger.warning("No flow segment data found in input")
            return {"error": "No flow segment data found"}
            
        # Extract the flow segment data
        segment = flow_data["flowSegmentData"]
        
        # Extract basic metrics
        current_speed = segment.get("currentSpeed")
        free_flow_speed = segment.get("freeFlowSpeed")
        current_tt = segment.get("currentTravelTime")
        free_flow_tt = segment.get("freeFlowTravelTime")
        confidence = segment.get("confidence")
        
        # Skip processing if essential data is missing
        if None in [current_speed, free_flow_speed]:
            logger.warning("Missing essential traffic flow data")
            return {"error": "Missing essential traffic data"}
            
        # Calculate additional metrics
        
        # Speed ratio (proportion of free flow speed)
        speed_ratio = current_speed / free_flow_speed if free_flow_speed > 0 else 0
        
        # Delay time in seconds
        delay = 0
        if current_tt is not None and free_flow_tt is not None:
            delay = max(0, current_tt - free_flow_tt)
            
        # Delay ratio (proportion of travel time that is delay)
        delay_ratio = delay / free_flow_tt if free_flow_tt and free_flow_tt > 0 else 0
        
        # Determine congestion level based on speed ratio
        congestion_level = self._classify_congestion(speed_ratio)
        
        # Calculate a traffic score (0-100, higher is better)
        traffic_score = self._calculate_traffic_score(speed_ratio, delay_ratio, confidence)
        
        # Create processed data dictionary with all metrics
        processed_data = {
            "current_speed": current_speed,
            "free_flow_speed": free_flow_speed, 
            "current_travel_time": current_tt,
            "free_flow_travel_time": free_flow_tt,
            "confidence": confidence,
            "road_closure": segment.get("roadClosure", False),
            "speed_ratio": round(speed_ratio, 2),
            "delay": delay,
            "delay_ratio": round(delay_ratio, 2),
            "congestion_level": congestion_level,
            "traffic_score": traffic_score,
            "timestamp": datetime.now().isoformat()
        }
        
        return processed_data

    def process_intersection_data(self, intersection_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process traffic data for an entire intersection with multiple approaches.
        
        Args:
            intersection_data (dict): Raw intersection data with approaches
            
        Returns:
            dict: Processed intersection data with metrics for each approach and overall scores
        """
        if "approaches" not in intersection_data:
            logger.warning("No approach data found in intersection data")
            return {"error": "No approach data found"}
            
        approaches = intersection_data["approaches"]
        processed_approaches = {}
        approach_scores = []
        approach_delays = []
        weighted_scores = []
        
        # Process each approach
        for direction, data in approaches.items():
            # Skip if there's an error with this approach data
            if "error" in data:
                processed_approaches[direction] = {"error": data["error"]}
                continue
                
            # Process the flow data for this approach
            processed_data = {}
            current_speed = data.get("current_speed")
            free_flow_speed = data.get("free_flow_speed")
            
            if current_speed is not None and free_flow_speed is not None:
                # Speed ratio
                speed_ratio = current_speed / free_flow_speed if free_flow_speed > 0 else 0
                
                # Delay calculation
                delay = data.get("delay", 0)
                free_flow_tt = data.get("free_flow_travel_time", 0)
                delay_ratio = delay / free_flow_tt if free_flow_tt and free_flow_tt > 0 else 0
                
                # Determine congestion level
                congestion_level = self._classify_congestion(speed_ratio)
                
                # Calculate traffic score
                traffic_score = self._calculate_traffic_score(
                    speed_ratio, 
                    delay_ratio, 
                    data.get("confidence", 1)
                )
                
                # Add derived metrics to the processed data
                processed_data = {
                    **data,  # Include all original data
                    "speed_ratio": round(speed_ratio, 2),
                    "delay_ratio": round(delay_ratio, 2) if free_flow_tt else 0,
                    "congestion_level": congestion_level,
                    "traffic_score": traffic_score
                }
                
                # Track scores and delays for intersection-level metrics
                approach_scores.append(traffic_score)
                approach_delays.append(delay if delay else 0)
                
                # Apply direction weights for weighted average
                weight = self.approach_weights.get(direction, 1.0)
                weighted_scores.append(traffic_score * weight)
            else:
                # Handle missing data
                processed_data = {
                    **data,
                    "error": "Incomplete data for processing"
                }
                
            # Store the processed approach data
            processed_approaches[direction] = processed_data
            
        # Calculate intersection-level metrics
        intersection_metrics = {}
        
        if approach_scores:
            # Calculate the overall health of the intersection
            average_score = sum(approach_scores) / len(approach_scores)
            weighted_avg_score = sum(weighted_scores) / sum(self.approach_weights.values())
            max_delay = max(approach_delays) if approach_delays else 0
            total_delay = sum(approach_delays)
            
            # Determine the most and least congested approaches
            approaches_list = [(dir, data.get("traffic_score", 0)) 
                              for dir, data in processed_approaches.items() 
                              if "traffic_score" in data]
            
            most_congested = min(approaches_list, key=lambda x: x[1])[0] if approaches_list else None
            least_congested = max(approaches_list, key=lambda x: x[1])[0] if approaches_list else None
            
            # Assign intersection-level metrics
            intersection_metrics = {
                "average_score": round(average_score, 1),
                "weighted_score": round(weighted_avg_score, 1),
                "max_delay": max_delay,
                "total_delay": total_delay,
                "most_congested_approach": most_congested,
                "least_congested_approach": least_congested,
                "number_of_approaches": len(approaches),
                "timestamp": datetime.now().isoformat()
            }
            
            # Determine overall congestion level for the intersection
            if weighted_avg_score >= 80:
                intersection_metrics["overall_status"] = "Good"
            elif weighted_avg_score >= 60:
                intersection_metrics["overall_status"] = "Moderate"
            elif weighted_avg_score >= 40:
                intersection_metrics["overall_status"] = "Congested"
            else:
                intersection_metrics["overall_status"] = "Severely Congested"
                
        # Combine everything into the final processed data
        processed_data = {
            "intersection": {
                "latitude": intersection_data["intersection"]["latitude"],
                "longitude": intersection_data["intersection"]["longitude"],
                "timestamp": datetime.now().isoformat(),
                **intersection_metrics
            },
            "approaches": processed_approaches
        }
        
        return processed_data

    def process_incidents(self, incidents_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process raw traffic incidents data.
        
        Args:
            incidents_data (dict): Raw incidents data from TomTom API
            
        Returns:
            list: List of processed incidents with additional metrics
        """
        processed_incidents = []
        
        # Return empty list if there's an error or no incidents
        if "error" in incidents_data:
            logger.warning(f"Error in incidents data: {incidents_data['error']}")
            return processed_incidents
            
        if "incidents" not in incidents_data:
            logger.warning("No incidents found in data")
            return processed_incidents
            
        # Process each incident
        for incident in incidents_data["incidents"]:
            # Extract basic incident information
            incident_type = incident.get("type", "Unknown")
            properties = incident.get("properties", {})
            
            # Get incident attributes
            description = properties.get("events", [{}])[0].get("description", "No description")
            delay = properties.get("delay", 0)
            icon_category = properties.get("iconCategory", "Unknown")
            from_location = properties.get("from", "Unknown")
            to_location = properties.get("to", "Unknown")
            
            # Convert timestamps if available
            start_time = properties.get("startTime")
            end_time = properties.get("endTime")
            
            # Calculate incident severity based on delay and category
            severity = self._calculate_incident_severity(delay, icon_category)
            
            # Format the processed incident
            processed_incident = {
                "type": incident_type,
                "description": description,
                "from": from_location,
                "to": to_location,
                "delay": delay,
                "category": icon_category,
                "severity": severity,
                "start_time": start_time,
                "end_time": end_time
            }
            
            processed_incidents.append(processed_incident)
            
        return processed_incidents

    def save_processed_data(self, processed_data: Dict[str, Any], 
                          filename: str = None) -> str:
        """
        Save processed data to a JSON file.
        
        Args:
            processed_data (dict): Processed traffic data
            filename (str, optional): Custom filename. If None, a timestamp-based name is used.
            
        Returns:
            str: Path to the saved file
        """
        # Generate a timestamp-based filename if none provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"traffic_data_{timestamp}.json"
            
        # Ensure the filename has .json extension
        if not filename.endswith(".json"):
            filename += ".json"
            
        # Create the full file path
        file_path = os.path.join(self.data_dir, filename)
        
        # Save the data to a JSON file
        with open(file_path, "w") as f:
            json.dump(processed_data, f, indent=2)
            
        logger.info(f"Saved processed data to {file_path}")
        return file_path

    def load_processed_data(self, file_path: str) -> Dict[str, Any]:
        """
        Load processed data from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file
            
        Returns:
            dict: Loaded processed data
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            logger.info(f"Loaded processed data from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return {"error": str(e)}

    def append_to_time_series(self, processed_data: Dict[str, Any], 
                            csv_file: str = "traffic_time_series.csv") -> str:
        """
        Append processed intersection data to a CSV file for time series analysis.
        
        Args:
            processed_data (dict): Processed intersection data
            csv_file (str): CSV filename to append to
            
        Returns:
            str: Path to the CSV file
        """
        # Ensure the data contains intersection metrics
        if "intersection" not in processed_data:
            logger.warning("No intersection data found for time series")
            return None
            
        # Create the full file path
        file_path = os.path.join(self.data_dir, csv_file)
        
        # Extract the intersection metrics
        intersection = processed_data["intersection"]
        approaches = processed_data.get("approaches", {})
        
        # Create a row for the time series
        row = {
            "timestamp": intersection.get("timestamp", datetime.now().isoformat()),
            "latitude": intersection.get("latitude"),
            "longitude": intersection.get("longitude"),
            "average_score": intersection.get("average_score"),
            "weighted_score": intersection.get("weighted_score"),
            "max_delay": intersection.get("max_delay"),
            "total_delay": intersection.get("total_delay"),
            "overall_status": intersection.get("overall_status")
        }
        
        # Add metrics for each approach (if available)
        for direction in ["North", "South", "East", "West", "Northeast", "Northwest", "Southeast", "Southwest"]:
            if direction in approaches:
                approach = approaches[direction]
                row[f"{direction}_speed"] = approach.get("current_speed")
                row[f"{direction}_congestion"] = approach.get("congestion_level")
                row[f"{direction}_delay"] = approach.get("delay")
                row[f"{direction}_score"] = approach.get("traffic_score")
        
        # Convert to DataFrame for easier CSV handling
        df = pd.DataFrame([row])
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.isfile(file_path)
        
        # Append to the CSV file
        df.to_csv(file_path, mode='a', header=not file_exists, index=False)
        
        logger.info(f"Appended data to time series file {file_path}")
        return file_path

    def calculate_optimal_cycle_times(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate optimal traffic signal cycle times based on processed traffic data.
        
        Args:
            processed_data (dict): Processed intersection data
            
        Returns:
            dict: Recommended signal timings for each approach
        """
        if "approaches" not in processed_data:
            logger.warning("No approach data found for cycle time calculation")
            return {"error": "Missing approach data"}
            
        approaches = processed_data["approaches"]
        
        # Default minimum and maximum phase times (in seconds)
        min_phase_time = 5  # Minimum green time
        max_phase_time = 60  # Maximum green time
        yellow_time = 3  # Standard yellow time
        all_red_time = 2  # All-red clearance interval
        
        # Group approaches by primary directions
        primary_approaches = {
            "North-South": ["North", "South"],
            "East-West": ["East", "West"]
        }
        
        # Calculate volume-to-capacity ratio proxy using delay and congestion
        v_c_ratios = {}
        for direction, data in approaches.items():
            if "error" in data:
                continue
                
            # Skip approaches without necessary data
            if "delay" not in data or "congestion_level" not in data:
                continue
                
            # Calculate a proxy for volume-to-capacity ratio
            # Higher delay and congestion level indicate higher V/C ratio
            congestion_value = {
                "Free Flow": 0.3,
                "Light": 0.5,
                "Moderate": 0.7,
                "Heavy": 0.85,
                "Severe": 0.95
            }.get(data["congestion_level"], 0.5)
            
            delay = data.get("delay", 0)
            delay_factor = min(1.0, delay / 180.0)  # Normalize delay, cap at 180 seconds
            
            # Combine congestion and delay factors
            v_c_ratios[direction] = (congestion_value * 0.7) + (delay_factor * 0.3)
        
        # Calculate group V/C ratios
        group_v_c_ratios = {}
        for group_name, directions in primary_approaches.items():
            valid_directions = [d for d in directions if d in v_c_ratios]
            if valid_directions:
                group_v_c_ratios[group_name] = max([v_c_ratios[d] for d in valid_directions])
            else:
                group_v_c_ratios[group_name] = 0.5  # Default if no valid directions
        
        # Determine total cycle length based on highest V/C ratio
        max_v_c = max(group_v_c_ratios.values()) if group_v_c_ratios else 0.5
        
        # Webster's formula (simplified): C = (1.5*L + 5) / (1 - Y)
        # where L is total lost time and Y is sum of critical flow ratios
        lost_time = (yellow_time + all_red_time) * len(primary_approaches)
        critical_sum = sum(group_v_c_ratios.values())
        
        # Prevent division by zero or negative values
        if critical_sum >= 0.95:
            critical_sum = 0.95
            
        cycle_length = (1.5 * lost_time + 5) / (1 - critical_sum)
        
        # Constrain cycle length to reasonable values
        cycle_length = max(30, min(120, cycle_length))
        cycle_length = round(cycle_length)  # Round to nearest second
        
        # Distribute green time proportionally to V/C ratios
        green_times = {}
        available_green_time = cycle_length - lost_time
        
        for group_name, v_c in group_v_c_ratios.items():
            if critical_sum > 0:  # Prevent division by zero
                proportion = v_c / critical_sum
                green_time = available_green_time * proportion
            else:
                green_time = available_green_time / len(group_v_c_ratios)
                
            # Ensure minimum green time
            green_time = max(min_phase_time, min(max_phase_time, green_time))
            green_times[group_name] = round(green_time)  # Round to nearest second
        
        # Adjust to ensure the total cycle length is maintained
        total_assigned = sum(green_times.values()) + lost_time
        if total_assigned != cycle_length:
            # Distribute the difference proportionally
            diff = cycle_length - total_assigned

            # Keep adjusting until the difference is distributed or we can't adjust anymore

            while abs(diff) > 0:
                adjusted = False
                for group in sorted(green_times.keys(), key=lambda k: green_times[k], reverse=(diff > 0)):
                    adjustment = 1 if diff > 0 else -1
                    if min_phase_time <= green_times[group] + adjustment <= max_phase_time:
                        green_times[group] += adjustment
                        diff -= adjustment
                        adjusted = True
                        if diff == 0:
                            break
        
                # If we couldn't make any adjustments in this iteration, break the loop
                if not adjusted:
                    break
        
        # Construct the final timing recommendations
        timing_recommendations = {
            "cycle_length": cycle_length,
            "lost_time_per_phase": yellow_time + all_red_time,
            "total_lost_time": lost_time,
            "max_v_c_ratio": max_v_c,
            "phase_times": {
                group: {
                    "green_time": green_time,
                    "yellow_time": yellow_time,
                    "all_red_time": all_red_time,
                    "total_phase_time": green_time + yellow_time + all_red_time
                }
                for group, green_time in green_times.items()
            }
        }
        
        return timing_recommendations

    def analyze_time_series(self, csv_file: str = "traffic_time_series.csv") -> Dict[str, Any]:
        """
        Analyze traffic time series data to identify patterns.
        
        Args:
            csv_file (str): CSV file containing time series data
            
        Returns:
            dict: Analysis results including trends and patterns
        """
        file_path = os.path.join(self.data_dir, csv_file)
        
        # Check if file exists
        if not os.path.isfile(file_path):
            logger.warning(f"Time series file not found: {file_path}")
            return {"error": "Time series file not found"}
            
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Extract time components
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
            
            # Initialize results dictionary
            results = {
                "data_points": len(df),
                "time_span": {
                    "start": df['timestamp'].min().isoformat(),
                    "end": df['timestamp'].max().isoformat()
                },
                "hourly_patterns": {},
                "daily_patterns": {},
                "congestion_hotspots": []
            }
            
            # Analyze hourly patterns for average score
            if 'weighted_score' in df.columns:
                hourly_avg = df.groupby('hour')['weighted_score'].mean()
                peak_hour = hourly_avg.idxmin()  # Lowest score = worst traffic
                best_hour = hourly_avg.idxmax()  # Highest score = best traffic
                
                results["hourly_patterns"] = {
                    "peak_congestion_hour": int(peak_hour),
                    "peak_congestion_score": round(hourly_avg[peak_hour], 1),
                    "least_congestion_hour": int(best_hour),
                    "least_congestion_score": round(hourly_avg[best_hour], 1),
                    "hourly_average_scores": {
                        str(hour): round(score, 1) 
                        for hour, score in hourly_avg.items()
                    }
                }
            
            # Analyze daily patterns
            if 'day_of_week' in df.columns and 'weighted_score' in df.columns:
                daily_avg = df.groupby('day_of_week')['weighted_score'].mean()
                
                day_names = {
                    0: "Monday", 1: "Tuesday", 2: "Wednesday", 
                    3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
                }
                
                results["daily_patterns"] = {
                    "worst_day": day_names[daily_avg.idxmin()],
                    "worst_day_score": round(daily_avg.min(), 1),
                    "best_day": day_names[daily_avg.idxmax()],
                    "best_day_score": round(daily_avg.max(), 1),
                    "daily_average_scores": {
                        day_names[day]: round(score, 1) 
                        for day, score in daily_avg.items()
                    }
                }
            
            # Find congestion hotspots (approaches that are frequently congested)
            approach_cols = [col for col in df.columns if col.endswith('_congestion')]
            if approach_cols:
                hotspots = []
                for col in approach_cols:
                    direction = col.split('_')[0]
                    
                    # Count occurrences of each congestion level
                    congestion_counts = df[col].value_counts(normalize=True)
                    
                    # Consider an approach a hotspot if it has Heavy or Severe congestion > 25% of the time
                    severe_pct = congestion_counts.get('Severe', 0) * 100
                    heavy_pct = congestion_counts.get('Heavy', 0) * 100
                    problem_pct = severe_pct + heavy_pct
                    
                    if problem_pct > 25:
                        hotspots.append({
                            "approach": direction,
                            "severe_congestion_percentage": round(severe_pct, 1),
                            "heavy_congestion_percentage": round(heavy_pct, 1),
                            "total_problem_percentage": round(problem_pct, 1)
                        })
                
                # Sort hotspots by problem percentage (highest first)
                hotspots.sort(key=lambda x: x["total_problem_percentage"], reverse=True)
                results["congestion_hotspots"] = hotspots
            
            # Calculate overall trends
            if 'weighted_score' in df.columns and len(df) > 1:
                # Simple linear trend (positive = improving, negative = worsening)
                df = df.sort_values('timestamp')
                scores = df['weighted_score'].values
                
                if len(scores) > 1:
                    # Calculate slope using simple linear regression
                    x = np.arange(len(scores))
                    slope = np.polyfit(x, scores, 1)[0]
                    
                    # Interpret the trend
                    if abs(slope) < 0.01:
                        trend = "Stable"
                    elif slope > 0:
                        trend = "Improving"
                    else:
                        trend = "Worsening"
                    
                    results["overall_trend"] = {
                        "direction": trend,
                        "slope": round(slope, 4),
                        "first_score": round(scores[0], 1),
                        "last_score": round(scores[-1], 1),
                        "change": round(scores[-1] - scores[0], 1)
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing time series data: {e}")
            return {"error": str(e)}

    def _classify_congestion(self, speed_ratio: float) -> str:
        """
        Classify the level of congestion based on the ratio of current speed to free flow speed.
        
        Args:
            speed_ratio (float): Ratio of current speed to free flow speed
            
        Returns:
            str: Congestion level classification
        """
        if speed_ratio >= self.congestion_thresholds["free_flow"]:
            return "Free Flow"
        elif speed_ratio >= self.congestion_thresholds["light"]:
            return "Light"
        elif speed_ratio >= self.congestion_thresholds["moderate"]:
            return "Moderate"
        elif speed_ratio >= self.congestion_thresholds["heavy"]:
            return "Heavy"
        else:
            return "Severe"

    def _calculate_traffic_score(self, speed_ratio: float, delay_ratio: float, 
                              confidence: float = 1.0) -> int:
        """
        Calculate a traffic health score (0-100, higher is better).
        
        Args:
            speed_ratio (float): Ratio of current speed to free flow speed
            delay_ratio (float): Ratio of delay to free flow travel time
            confidence (float): Data confidence factor (0-1)
            
        Returns:
            int: Traffic health score
        """
        # Convert ratios to scores (0-100)
        speed_score = min(100, max(0, speed_ratio * 100))
        delay_score = min(100, max(0, (1 - delay_ratio) * 100))
        
        # Weight the factors (speed is more important than delay)
        weighted_score = (speed_score * 0.7) + (delay_score * 0.3)
        
        # Apply confidence factor
        confidence = max(0.1, min(1.0, confidence))  # Ensure confidence is between 0.1 and 1.0
        final_score = weighted_score * confidence
        
        return round(final_score)

    def _calculate_incident_severity(self, delay: Optional[int], 
                                  category: str) -> str:
        """
        Calculate the severity of a traffic incident.
        
        Args:
            delay (int): Delay in seconds caused by the incident
            category (str): Incident category from TomTom API
            
        Returns:
            str: Severity classification
        """
        # Default severity based on delay (if available)
        if delay is None:
            delay = 0
            
        if delay > 900:  # > 15 minutes
            severity = "Severe"
        elif delay > 600:  # > 10 minutes
            severity = "Major"
        elif delay > 300:  # > 5 minutes
            severity = "Moderate"
        elif delay > 60:  # > 1 minute
            severity = "Minor"
        else:
            severity = "Low"
        
        # Adjust based on category if available
        if category:
            category = str(category).lower()
            
            # Override for certain high-impact categories
            if "accident" in category or "road closure" in category:
                if severity in ["Low", "Minor"]:
                    severity = "Moderate"
            elif "construction" in category:
                if severity == "Low":
                    severity = "Minor"
                    
        return severity
        
# Example usage
if __name__ == "__main__":
    # Example traffic flow data
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
    
    # Create data processor
    processor = DataProcessor()
    
    # Process the sample data
    processed_flow = processor.process_traffic_flow(sample_flow_data)
    print("Processed Flow Data:")
    print(json.dumps(processed_flow, indent=2))
    
    # Calculate optimal cycle times
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
    
    # Process the intersection data
    processed_intersection = processor.process_intersection_data(sample_intersection_data)
    print("\nProcessed Intersection Data:")
    print(json.dumps(processed_intersection, indent=2))
    
    # Calculate optimal signal timings
    timings = processor.calculate_optimal_cycle_times(processed_intersection)
    print("\nOptimal Signal Timings:")
    print(json.dumps(timings, indent=2))
    
    # Save the processed data
    processor.save_processed_data(processed_intersection, "sample_intersection.json")
    
    # Append to time series for future analysis
    processor.append_to_time_series(processed_intersection)