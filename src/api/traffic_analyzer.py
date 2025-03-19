"""
Traffic Analyzer

This module provides advanced traffic analysis capabilities including pattern detection,
prediction of future traffic conditions, and optimization of traffic signal plans.
It builds upon the data collected by the TomTomClient and processed by the DataProcessor.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrafficAnalyzer:
    """
    Provides advanced traffic analysis and prediction capabilities.
    """
    
    def __init__(self, data_dir: str = "./data", model_dir: str = "./models"):
        """
        Initialize the traffic analyzer.
        
        Args:
            data_dir (str): Directory containing processed traffic data
            model_dir (str): Directory to store trained prediction models
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # Create directories if they don't exist
        for directory in [data_dir, model_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
        
        # Default window sizes for different time periods
        self.time_windows = {
            "hour": 4,      # 4 data points per hour (15-minute intervals)
            "day": 96,      # 96 data points per day (4 per hour * 24 hours)
            "week": 672     # 672 data points per week (96 per day * 7 days)
        }
        
        # Load any existing prediction models
        self.prediction_models = {}
        self._try_load_models()
        
    def _try_load_models(self):
        """Attempt to load any previously trained prediction models."""
        if os.path.exists(self.model_dir):
            model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.json')]
            for model_file in model_files:
                try:
                    model_path = os.path.join(self.model_dir, model_file)
                    with open(model_path, 'r') as f:
                        model_info = json.load(f)
                    
                    # Initialize an appropriate model based on the saved info
                    model_type = model_info.get('model_type')
                    intersection_id = model_info.get('intersection_id')
                    
                    if model_type and intersection_id:
                        # Create a unique key for this model
                        model_key = f"{intersection_id}_{model_type}"
                        self.prediction_models[model_key] = model_info
                        logger.info(f"Loaded model info for {model_key}")
                except Exception as e:
                    logger.warning(f"Failed to load model from {model_file}: {e}")
    
    def load_time_series_data(self, csv_file: str = "traffic_time_series.csv") -> pd.DataFrame:
        """
        Load traffic time series data from a CSV file.
        
        Args:
            csv_file (str): CSV file containing traffic time series data
            
        Returns:
            DataFrame: Loaded time series data or None if file not found
        """
        file_path = os.path.join(self.data_dir, csv_file)
        
        if not os.path.isfile(file_path):
            logger.warning(f"Time series file not found: {file_path}")
            return None
            
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Extract time components for analysis
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            logger.info(f"Loaded time series data with {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading time series data: {e}")
            return None
    
    def detect_recurring_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect recurring traffic patterns in time series data.
        
        Args:
            df (DataFrame): Traffic time series data
            
        Returns:
            dict: Dictionary of detected patterns
        """
        if df is None or len(df) < 24:  # Need at least 24 hours of data
            logger.warning("Insufficient data for pattern detection")
            return {"error": "Insufficient data"}
        
        try:
            # Initialize results dictionary
            patterns = {
                "daily_patterns": {},
                "hourly_patterns": {},
                "day_of_week_patterns": {},
                "weekend_vs_weekday": {}
            }
            
            # Analyze hourly patterns
            hourly_avg = df.groupby('hour')[['weighted_score']].mean()
            patterns["hourly_patterns"] = {
                "peak_hours": list(hourly_avg.nsmallest(2, 'weighted_score').index),
                "off_peak_hours": list(hourly_avg.nlargest(2, 'weighted_score').index),
                "hourly_averages": hourly_avg['weighted_score'].to_dict()
            }
            
            # Analyze day of week patterns
            if len(df['day_of_week'].unique()) >= 5:  # Need at least 5 days of data
                day_names = {
                    0: "Monday", 1: "Tuesday", 2: "Wednesday", 
                    3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
                }
                
                daily_avg = df.groupby('day_of_week')[['weighted_score']].mean()
                patterns["day_of_week_patterns"] = {
                    "worst_day": day_names[daily_avg['weighted_score'].idxmin()],
                    "best_day": day_names[daily_avg['weighted_score'].idxmax()],
                    "daily_averages": {day_names[day]: score for day, score in 
                                    daily_avg['weighted_score'].items()}
                }
                
                # Weekend vs weekday comparison
                weekday_avg = df[df['is_weekend'] == 0]['weighted_score'].mean()
                weekend_avg = df[df['is_weekend'] == 1]['weighted_score'].mean()
                
                patterns["weekend_vs_weekday"] = {
                    "weekday_average": weekday_avg,
                    "weekend_average": weekend_avg,
                    "difference_percentage": ((weekend_avg - weekday_avg) / weekday_avg) * 100
                }
            
            # Look for recurring patterns - correlations between days/times
            if len(df) >= 7 * 24:  # At least one week of data
                # Create a pivot table with hours as columns and days as rows
                pivot = df.pivot_table(
                    index='day_of_week', 
                    columns='hour', 
                    values='weighted_score',
                    aggfunc='mean'
                )
                
                # Calculate correlation matrix between days
                day_correlations = pivot.T.corr()
                
                # Find the most similar days
                similar_days = []
                for i in range(7):
                    for j in range(i+1, 7):
                        if day_correlations.iloc[i, j] > 0.7:  # Strong correlation
                            similar_days.append((day_names[i], day_names[j], day_correlations.iloc[i, j]))
                
                patterns["similar_days"] = [
                    {"day1": day1, "day2": day2, "correlation": round(corr, 2)} 
                    for day1, day2, corr in similar_days
                ]
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return {"error": str(e)}
    
    def predict_future_conditions(self, intersection_id: str, df: pd.DataFrame, 
                                hours_ahead: int = 24) -> Dict[str, Any]:
        """
        Predict future traffic conditions for an intersection.
        
        Args:
            intersection_id (str): Unique identifier for the intersection
            df (DataFrame): Historical traffic time series data
            hours_ahead (int): Number of hours ahead to predict
            
        Returns:
            dict: Predicted traffic conditions
        """
        if df is None or len(df) < 48:  # Need at least 48 hours of data
            logger.warning("Insufficient data for prediction")
            return {"error": "Insufficient data for prediction"}
            
        try:
            # Prepare the data for prediction
            # We'll use previous traffic conditions, time of day, and day of week
            # to predict future traffic
            
            # Feature engineering
            X = pd.DataFrame()
            y = df['weighted_score'].values
            
            # Add time features
            X['hour'] = df['hour']
            X['day_of_week'] = df['day_of_week']
            X['is_weekend'] = df['is_weekend']
            
            # Add lagged values (t-1, t-2, t-3, t-4) for traffic scores
            for i in range(1, 5):
                X[f'score_lag_{i}'] = df['weighted_score'].shift(i)
                
            # Drop rows with NaN values
            X = X.dropna()
            y = y[4:]  # Match the length of X after dropping NaN rows
            
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Build and train the model
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            val_score = model.score(X_val, y_val)
            logger.info(f"Model validation R² score: {val_score:.4f}")
            
            # Generate future data points
            last_timestamp = df['timestamp'].max()
            predictions = []
            
            # Get the most recent feature values
            latest_features = X.iloc[-1].copy()
            
            # Generate predictions for each hour ahead
            for i in range(1, hours_ahead + 1):
                # Update time features
                future_time = last_timestamp + timedelta(hours=i)
                latest_features['hour'] = future_time.hour
                latest_features['day_of_week'] = future_time.dayofweek
                latest_features['is_weekend'] = 1 if future_time.dayofweek >= 5 else 0
                
                # Make prediction
                pred_score = model.predict([latest_features])[0]
                
                # Update lagged values for next prediction
                for j in range(4, 1, -1):
                    latest_features[f'score_lag_{j}'] = latest_features[f'score_lag_{j-1}']
                latest_features['score_lag_1'] = pred_score
                
                # Map predicted score to a congestion level
                congestion_level = self._score_to_congestion_level(pred_score)
                
                # Add to predictions list
                predictions.append({
                    "timestamp": future_time.isoformat(),
                    "predicted_score": round(float(pred_score), 1),
                    "congestion_level": congestion_level
                })
            
            # Save model info
            model_info = {
                "model_type": "traffic_prediction",
                "intersection_id": intersection_id,
                "trained_at": datetime.now().isoformat(),
                "validation_score": val_score,
                "feature_importance": dict(zip(X.columns, model.named_steps['regressor'].feature_importances_))
            }
            
            model_key = f"{intersection_id}_traffic_prediction"
            self.prediction_models[model_key] = model_info
            
            # Save model info to file
            model_file = os.path.join(self.model_dir, f"{model_key}.json")
            with open(model_file, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            return {
                "model_info": model_info,
                "predictions": predictions
            }
            
        except Exception as e:
            logger.error(f"Error predicting future conditions: {e}")
            return {"error": str(e)}
    
    def optimize_signal_plan(self, intersection_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize traffic signal timing plans based on predicted conditions.
        
        Args:
            intersection_id (str): Unique identifier for the intersection
            df (DataFrame): Historical traffic time series data
            
        Returns:
            dict: Optimized signal timing plans for different time periods
        """
        if df is None or len(df) < 24:  # Need at least 24 hours of data
            logger.warning("Insufficient data for signal optimization")
            return {"error": "Insufficient data for signal optimization"}
            
        try:
            # Group data by hour of day and day type (weekday/weekend)
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
            
            grouped_data = df.groupby(['hour_of_day', 'is_weekend']).agg({
                'North_speed': 'mean',
                'South_speed': 'mean',
                'East_speed': 'mean',
                'West_speed': 'mean',
                'North_delay': 'mean',
                'South_delay': 'mean',
                'East_delay': 'mean',
                'West_delay': 'mean',
                'weighted_score': 'mean'
            }).reset_index()
            
            # Define time periods
            time_periods = [
                {"name": "Early Morning", "hours": list(range(5, 7)), "description": "5:00 AM - 6:59 AM"},
                {"name": "AM Peak", "hours": list(range(7, 10)), "description": "7:00 AM - 9:59 AM"},
                {"name": "Midday", "hours": list(range(10, 16)), "description": "10:00 AM - 3:59 PM"},
                {"name": "PM Peak", "hours": list(range(16, 19)), "description": "4:00 PM - 6:59 PM"},
                {"name": "Evening", "hours": list(range(19, 23)), "description": "7:00 PM - 10:59 PM"},
                {"name": "Late Night", "hours": list(range(23, 24)) + list(range(0, 5)), "description": "11:00 PM - 4:59 AM"}
            ]
            
            # Optimize signal timing for each time period and day type
            optimized_plans = {}
            
            for day_type in ["Weekday", "Weekend"]:
                is_weekend = 1 if day_type == "Weekend" else 0
                optimized_plans[day_type] = {}
                
                for period in time_periods:
                    period_data = grouped_data[
                        (grouped_data['hour_of_day'].isin(period['hours'])) &
                        (grouped_data['is_weekend'] == is_weekend)
                    ]
                    
                    if len(period_data) == 0:
                        # No data for this period
                        continue
                        
                    # Calculate average metrics for this period
                    avg_data = period_data.mean()
                    
                    # Calculate V/C ratio proxies based on speeds and delays
                    north_south_delay = (avg_data['North_delay'] + avg_data['South_delay']) / 2
                    east_west_delay = (avg_data['East_delay'] + avg_data['West_delay']) / 2
                    
                    total_delay = north_south_delay + east_west_delay
                    
                    # Calculate proportions
                    if total_delay > 0:
                        north_south_proportion = north_south_delay / total_delay
                        east_west_proportion = east_west_delay / total_delay
                    else:
                        # Equal split if no delay data
                        north_south_proportion = 0.5
                        east_west_proportion = 0.5
                    
                    # Determine appropriate cycle length based on congestion
                    avg_score = avg_data['weighted_score']
                    
                    if avg_score >= 75:
                        # Good traffic conditions - shorter cycle
                        cycle_length = 80
                    elif avg_score >= 50:
                        # Moderate traffic - medium cycle
                        cycle_length = 100
                    else:
                        # Heavy traffic - longer cycle
                        cycle_length = 120
                    
                    # Calculate lost time
                    yellow_time = 3
                    all_red_time = 2
                    lost_time = (yellow_time + all_red_time) * 2  # Two phases
                    
                    # Distribute green time
                    available_green_time = cycle_length - lost_time
                    
                    north_south_green = max(10, round(available_green_time * north_south_proportion))
                    east_west_green = max(10, round(available_green_time * east_west_proportion))
                    
                    # Adjust to maintain cycle length
                    total_green = north_south_green + east_west_green
                    if total_green != available_green_time:
                        # Distribute the difference
                        diff = available_green_time - total_green
                        if north_south_proportion >= east_west_proportion:
                            north_south_green += diff
                        else:
                            east_west_green += diff
                    
                    # Create plan
                    plan = {
                        "period_name": period['name'],
                        "description": period['description'],
                        "cycle_length": cycle_length,
                        "lost_time": lost_time,
                        "phases": {
                            "North-South": {
                                "green_time": north_south_green,
                                "yellow_time": yellow_time,
                                "all_red_time": all_red_time,
                                "total_phase_time": north_south_green + yellow_time + all_red_time
                            },
                            "East-West": {
                                "green_time": east_west_green,
                                "yellow_time": yellow_time,
                                "all_red_time": all_red_time,
                                "total_phase_time": east_west_green + yellow_time + all_red_time
                            }
                        },
                        "metrics": {
                            "average_score": round(avg_score, 1),
                            "north_south_delay": round(north_south_delay, 1),
                            "east_west_delay": round(east_west_delay, 1)
                        }
                    }
                    
                    optimized_plans[day_type][period['name']] = plan
            
            return {
                "intersection_id": intersection_id,
                "generated_at": datetime.now().isoformat(),
                "signal_plans": optimized_plans
            }
            
        except Exception as e:
            logger.error(f"Error optimizing signal plan: {e}")
            return {"error": str(e)}
    
    def visualize_traffic_patterns(self, df: pd.DataFrame, output_dir: str = "./visualizations"):
        """
        Generate visualizations of traffic patterns.
        
        Args:
            df (DataFrame): Traffic time series data
            output_dir (str): Directory to save visualizations
            
        Returns:
            list: Paths to generated visualization files
        """
        if df is None or len(df) < 24:
            logger.warning("Insufficient data for visualization")
            return []
            
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        visualization_files = []
        
        try:
            # 1. Daily traffic score patterns
            plt.figure(figsize=(12, 6))
            hourly_avg = df.groupby('hour')['weighted_score'].mean()
            hourly_std = df.groupby('hour')['weighted_score'].std()
            
            plt.plot(hourly_avg.index, hourly_avg.values, 'b-', linewidth=2)
            plt.fill_between(
                hourly_avg.index, 
                hourly_avg.values - hourly_std.values,
                hourly_avg.values + hourly_std.values,
                alpha=0.2, color='b'
            )
            
            plt.title('Average Traffic Score by Hour of Day')
            plt.xlabel('Hour of Day')
            plt.ylabel('Traffic Score (Higher is Better)')
            plt.xticks(range(0, 24, 2))
            plt.grid(True, linestyle='--', alpha=0.7)
            
            hourly_file = os.path.join(output_dir, 'hourly_traffic_pattern.png')
            plt.savefig(hourly_file)
            plt.close()
            
            visualization_files.append(hourly_file)
            
            # 2. Weekly pattern heatmap
            if len(df['day_of_week'].unique()) >= 5:
                plt.figure(figsize=(12, 8))
                
                # Create a pivot table with hours as columns and days as rows
                pivot = df.pivot_table(
                    index='day_of_week', 
                    columns='hour', 
                    values='weighted_score',
                    aggfunc='mean'
                )
                
                # Map numeric days to names
                day_names = {
                    0: "Monday", 1: "Tuesday", 2: "Wednesday", 
                    3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
                }
                
                pivot.index = [day_names[day] for day in pivot.index]
                
                # Create heatmap
                plt.imshow(
                    pivot, 
                    cmap='YlGnBu_r',  # Reversed YlGnBu - darker means worse traffic
                    aspect='auto',
                    interpolation='nearest'
                )
                
                # Add colorbar
                cbar = plt.colorbar()
                cbar.set_label('Traffic Score (Higher is Better)')
                
                # Add labels and ticks
                plt.title('Weekly Traffic Pattern Heatmap')
                plt.xlabel('Hour of Day')
                plt.ylabel('Day of Week')
                plt.xticks(range(0, 24, 2), range(0, 24, 2))
                plt.yticks(range(len(pivot.index)), pivot.index)
                
                weekly_file = os.path.join(output_dir, 'weekly_traffic_heatmap.png')
                plt.savefig(weekly_file)
                plt.close()
                
                visualization_files.append(weekly_file)
                
            # 3. Approach-specific patterns
            approach_cols = [col for col in df.columns if col.endswith('_speed')]
            
            if approach_cols:
                plt.figure(figsize=(12, 6))
                
                for col in approach_cols:
                    direction = col.split('_')[0]
                    hourly_avg = df.groupby('hour')[col].mean()
                    plt.plot(hourly_avg.index, hourly_avg.values, linewidth=2, label=direction)
                
                plt.title('Average Speed by Approach and Hour of Day')
                plt.xlabel('Hour of Day')
                plt.ylabel('Speed (km/h)')
                plt.xticks(range(0, 24, 2))
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                
                approaches_file = os.path.join(output_dir, 'approach_speeds.png')
                plt.savefig(approaches_file)
                plt.close()
                
                visualization_files.append(approaches_file)
            
            return visualization_files
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            return []
    
    def _score_to_congestion_level(self, traffic_score: float) -> str:
        """
        Convert a traffic score to a congestion level.
        
        Args:
            traffic_score (float): Traffic score (0-100, higher is better)
            
        Returns:
            str: Congestion level
        """
        if traffic_score >= 80:
            return "Free Flow"
        elif traffic_score >= 60:
            return "Light"
        elif traffic_score >= 40:
            return "Moderate"
        elif traffic_score >= 20:
            return "Heavy"
        else:
            return "Severe"
    
    def analyze_intersection_performance(self, intersection_id: str, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of an intersection's performance.
        
        Args:
            intersection_id (str): Unique identifier for the intersection
            df (DataFrame, optional): Traffic time series data. If None, will load from default file.
            
        Returns:
            dict: Comprehensive analysis results
        """
        # Load data if not provided
        if df is None:
            df = self.load_time_series_data()
            
        if df is None or len(df) < 24:
            logger.warning("Insufficient data for intersection analysis")
            return {"error": "Insufficient data for analysis"}
            
        try:
            # Perform analysis steps
            
            # 1. Detect recurring patterns
            patterns = self.detect_recurring_patterns(df)
            
            # 2. Predict future conditions (next 24 hours)
            predictions = self.predict_future_conditions(intersection_id, df)
            
            # 3. Optimize signal plans
            signal_plans = self.optimize_signal_plan(intersection_id, df)
            
            # 4. Generate visualizations
            visualization_files = self.visualize_traffic_patterns(df)
            
            # Compile the comprehensive analysis
            analysis = {
                "intersection_id": intersection_id,
                "analysis_timestamp": datetime.now().isoformat(),
                "data_points_analyzed": len(df),
                "time_span": {
                    "start": df['timestamp'].min().isoformat(),
                    "end": df['timestamp'].max().isoformat()
                },
                "traffic_patterns": patterns,
                "future_predictions": predictions.get("predictions", []),
                "optimized_signal_plans": signal_plans.get("signal_plans", {}),
                "visualization_files": visualization_files,
                "overall_findings": self._generate_overall_findings(df, patterns)
            }
            
            # Save the analysis to a file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_file = os.path.join(self.data_dir, f"analysis_{intersection_id}_{timestamp}.json")
            
            with open(analysis_file, 'w') as f:
                # Need a custom JSON encoder to handle non-serializable objects
                json.dump(analysis, f, indent=2, default=str)
                
            logger.info(f"Saved comprehensive analysis to {analysis_file}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing intersection performance: {e}")
            return {"error": str(e)}
    
    def _generate_overall_findings(self, df: pd.DataFrame, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate overall findings and recommendations based on the analysis.
        
        Args:
            df (DataFrame): Traffic time series data
            patterns (dict): Detected traffic patterns
            
        Returns:
            dict: Overall findings and recommendations
        """
        findings = {
            "key_issues": [],
            "recommendations": []
        }
        
        # Check for peak congestion periods
        if "hourly_patterns" in patterns and "peak_hours" in patterns["hourly_patterns"]:
            peak_hours = patterns["hourly_patterns"]["peak_hours"]
            findings["key_issues"].append(
                f"Highest congestion occurs at hours: {', '.join(map(str, peak_hours))}"
            )
            
            # Recommend coordinated signal timing during peak hours
            findings["recommendations"].append(
                f"Implement specialized timing plans during peak hours: {', '.join(map(str, peak_hours))}"
            )
        
        # Check for day-of-week patterns
        if "day_of_week_patterns" in patterns and "worst_day" in patterns["day_of_week_patterns"]:
            worst_day = patterns["day_of_week_patterns"]["worst_day"]
            findings["key_issues"].append(
                f"Highest weekly congestion occurs on {worst_day}"
            )
        
        # Check for weekend vs weekday differences
        if "weekend_vs_weekday" in patterns and "difference_percentage" in patterns["weekend_vs_weekday"]:
            diff_pct = patterns["weekend_vs_weekday"]["difference_percentage"]
            if abs(diff_pct) > 15:  # Significant difference
                better_period = "weekends" if diff_pct > 0 else "weekdays"
                worse_period = "weekdays" if diff_pct > 0 else "weekends"
                findings["key_issues"].append(
                    f"Traffic conditions are {abs(diff_pct):.1f}% better on {better_period} than on {worse_period}"
                )
                findings["recommendations"].append(
                    f"Implement separate signal timing plans for weekdays and weekends"
                )
        
        # Analyze approach-specific issues
        approach_cols = [col for col in df.columns if col.endswith('_congestion')]
        if approach_cols:
            problematic_approaches = []
            
            for col in approach_cols:
                direction = col.split('_')[0]
                
                # Count severe congestion occurrences
                severe_count = sum(df[col] == "Severe")
                heavy_count = sum(df[col] == "Heavy")
                problem_pct = (severe_count + heavy_count) / len(df) * 100
                
                if problem_pct > 25:  # More than 25% of the time has heavy/severe congestion
                    problematic_approaches.append({
                        "direction": direction,
                        "problem_percentage": round(problem_pct, 1)
                    })
            
            if problematic_approaches:
                # Sort by problem percentage (highest first)
                problematic_approaches.sort(key=lambda x: x["problem_percentage"], reverse=True)
                
                # Add to findings
                directions = [f"{a['direction']} ({a['problem_percentage']}%)" for a in problematic_approaches[:2]]
                findings["key_issues"].append(
                    f"Persistent congestion on approaches: {', '.join(directions)}"
                )
                
                # Recommend approach-specific improvements
                worst_approach = problematic_approaches[0]["direction"]
                findings["recommendations"].append(
                    f"Prioritize the {worst_approach} approach during signal timing optimization"
                )
                
                if len(problematic_approaches) > 1:
                    findings["recommendations"].append(
                        f"Consider physical improvements (lane additions, turn lanes) for the most congested approaches"
                    )
        
        # Overall assessment
        if df['weighted_score'].mean() < 50:
            findings["key_issues"].append(
                "Overall poor traffic performance across the analysis period"
            )
            findings["recommendations"].append(
                "Consider comprehensive corridor improvements including adaptive signal control"
            )
        
        return findings


# Example usage
if __name__ == "__main__":
    analyzer = TrafficAnalyzer()
    
    # Load time series data
    df = analyzer.load_time_series_data()
    
    if df is not None:
        # Perform pattern detection
        patterns = analyzer.detect_recurring_patterns(df)
        print("\nDetected Patterns:")
        print(json.dumps(patterns, indent=2))
        
        # Predict future conditions
        intersection_id = "intersection_123"  # Example ID
        predictions = analyzer.predict_future_conditions(intersection_id, df)
        
        if "error" not in predictions:
            print("\nPredicted Conditions (next 24 hours):")
            for pred in predictions["predictions"][:5]:  # Show first 5 predictions
                print(f"{pred['timestamp']}: Score={pred['predicted_score']}, Level={pred['congestion_level']}")
            
        # Visualize patterns
        visualization_files = analyzer.visualize_traffic_patterns(df)
        print(f"\nGenerated {len(visualization_files)} visualization files")
        
        # Perform comprehensive analysis
        analysis = analyzer.analyze_intersection_performance(intersection_id, df)
        print("\nAnalysis complete - results saved to file")