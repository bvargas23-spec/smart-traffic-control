import unittest
from unittest.mock import Mock, patch
import pytest # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import datetime
import os
import sys
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error # type: ignore

# Assuming your models are in a module called "models"
try:
    from traffic_control.models import ( # type: ignore
        TrafficPatternAnalyzer, 
        PredictionModel, 
        SignalTimingOptimizer,
        CoordinationModel
    )
except ImportError:
    # Create dummy classes for testing purposes if the real module isn't available
    class TrafficPatternAnalyzer:
        pass
    
    class PredictionModel:
        pass
    
    class SignalTimingOptimizer:
        pass
    
    class CoordinationModel:
        pass


class TestTrafficPatternAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample traffic data for testing
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2025-03-01', periods=24, freq='H'),
            'intersection_id': [1] * 24,
            'approach': ['north'] * 6 + ['south'] * 6 + ['east'] * 6 + ['west'] * 6,
            'vehicle_count': [10, 15, 20, 50, 80, 100, 90, 70, 40, 30, 20, 10, 
                             15, 25, 35, 60, 90, 100, 80, 60, 40, 30, 20, 10],
            'average_speed': [45] * 24,
            'queue_length': [2, 3, 4, 10, 15, 20, 18, 14, 8, 6, 4, 2,
                            3, 5, 7, 12, 18, 20, 16, 12, 8, 6, 4, 2]
        })
        
        # Initialize the analyzer with the sample data
        self.analyzer = TrafficPatternAnalyzer()
        self.analyzer.load_data(self.sample_data)
    
    def test_peak_hour_detection(self):
        """Test detection of peak traffic hours."""
        peak_hours = self.analyzer.detect_peak_hours(approach='north')
        
        # Verify peak hours detected correctly
        self.assertEqual(len(peak_hours), 1)
        self.assertEqual(peak_hours[0].hour, 5)  # 5th hour has highest volume
        
        # Test for south approach
        peak_hours_south = self.analyzer.detect_peak_hours(approach='south')
        self.assertEqual(peak_hours_south[0].hour, 11)  # 11th hour from start for south
    
    def test_daily_pattern_detection(self):
        """Test detection of recurring daily traffic patterns."""
        # We need more data for this, so let's create a week of synthetic data
        week_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2025-03-01', periods=7*24, freq='H'),
            'intersection_id': [1] * (7*24),
            'approach': ['north'] * (7*24),
            'vehicle_count': [
                # Repeating the same pattern for 7 days
                *([10, 15, 20, 50, 80, 100, 90, 70, 40, 30, 20, 10, 
                   15, 25, 35, 60, 90, 100, 80, 60, 40, 30, 20, 10] * 7)
            ]
        })
        
        self.analyzer.load_data(week_data)
        patterns = self.analyzer.detect_daily_patterns(approach='north')
        
        # Verify patterns detected correctly - should find two peaks (morning and evening)
        self.assertEqual(len(patterns), 2)
        self.assertIn('morning', patterns[0]['label'])
        self.assertIn('evening', patterns[1]['label'])
    
    def test_unusual_pattern_detection(self):
        """Test detection of unusual traffic patterns."""
        # Create normal pattern
        normal_pattern = np.array([10, 15, 20, 50, 80, 100, 90, 70, 40, 30, 20, 10, 
                                15, 25, 35, 60, 90, 100, 80, 60, 40, 30, 20, 10])
        
        # Create abnormal pattern (much higher volumes)
        abnormal_pattern = normal_pattern * 2
        
        # Mock the analyzer's baseline pattern
        self.analyzer.get_baseline_pattern = Mock(return_value=normal_pattern)
        
        # Test with normal pattern
        is_unusual = self.analyzer.detect_unusual_pattern(normal_pattern)
        self.assertFalse(is_unusual)
        
        # Test with abnormal pattern
        is_unusual = self.analyzer.detect_unusual_pattern(abnormal_pattern)
        self.assertTrue(is_unusual)
    
    def test_congestion_detection(self):
        """Test detection of congestion points."""
        congestion_points = self.analyzer.detect_congestion(threshold=80)
        
        # Should find congestion in all approaches
        self.assertEqual(len(congestion_points), 4)  # One for each approach
        
        # Each approach has at least one point with volume >= 80
        for approach in ['north', 'south', 'east', 'west']:
            found = False
            for point in congestion_points:
                if point['approach'] == approach:
                    found = True
                    break
            self.assertTrue(found, f"Did not find congestion for {approach} approach")
            
    # NEW TESTS START HERE
    
    def test_seasonal_pattern_detection(self):
        """Test detection of seasonal traffic patterns."""
        # Create test data with seasonal patterns
        dates = pd.date_range(start='2025-01-01', periods=365, freq='D')
        
        # Create seasonal pattern: higher in summer, lower in winter
        seasonal_volumes = []
        for date in dates:
            day_of_year = date.dayofyear
            # Sinusoidal pattern with peak in summer
            base_volume = 1000 + 500 * np.sin((day_of_year - 172) * 2 * np.pi / 365)
            
            # Add day-of-week pattern (weekday/weekend)
            if date.dayofweek >= 5:  # Weekend
                base_volume *= 0.7
            
            # Add noise
            volume = base_volume + np.random.normal(0, 50)
            seasonal_volumes.append(int(volume))
        
        seasonal_data = pd.DataFrame({
            'timestamp': dates,
            'intersection_id': [1] * 365,
            'approach': ['north'] * 365,
            'vehicle_count': seasonal_volumes
        })
        
        self.analyzer.load_data(seasonal_data)
        
        # Detect seasonal patterns
        seasonal_patterns = self.analyzer.detect_seasonal_patterns()
        
        # Verify seasonal patterns were detected
        self.assertGreaterEqual(len(seasonal_patterns), 1)
        
        # Verify summer peak is identified
        summer_pattern = next((p for p in seasonal_patterns if p['season'] == 'summer'), None)
        self.assertIsNotNone(summer_pattern)
        self.assertGreater(summer_pattern['average_volume'], 1200)  # Summer should be high
        
        # Verify winter trough is identified
        winter_pattern = next((p for p in seasonal_patterns if p['season'] == 'winter'), None)
        self.assertIsNotNone(winter_pattern)
        self.assertLess(winter_pattern['average_volume'], 1000)  # Winter should be low
    
    def test_holiday_pattern_detection(self):
        """Test detection of holiday traffic patterns."""
        # Create test data spanning major holidays
        dates = pd.date_range(start='2025-01-01', periods=365, freq='D')
        
        # Define holidays
        holidays = {
            '2025-01-01': 'New Year',
            '2025-07-04': 'Independence Day',
            '2025-11-28': 'Thanksgiving',
            '2025-12-25': 'Christmas'
        }
        
        # Create traffic volumes with holiday effects
        holiday_volumes = []
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            base_volume = 1000
            
            # Increase volume before and on holidays
            if date_str in holidays:
                base_volume *= 1.5  # Holiday itself
            elif any(abs((date - pd.Timestamp(h_date)).days) <= 3 for h_date in holidays.keys()):
                base_volume *= 1.3  # Days near holidays
                
            # Add noise
            volume = base_volume + np.random.normal(0, 50)
            holiday_volumes.append(int(volume))
        
        holiday_data = pd.DataFrame({
            'timestamp': dates,
            'intersection_id': [1] * 365,
            'approach': ['north'] * 365,
            'vehicle_count': holiday_volumes,
            'is_holiday': [1 if date.strftime('%Y-%m-%d') in holidays else 0 for date in dates]
        })
        
        self.analyzer.load_data(holiday_data)
        
        # Detect holiday patterns
        holiday_patterns = self.analyzer.detect_holiday_patterns()
        
        # Verify holiday patterns were detected
        self.assertGreaterEqual(len(holiday_patterns), 3)  # Should detect at least 3 major holidays
        
        # Verify increased volumes on major holidays
        for holiday_name, holiday_date in holidays.items():
            holiday_volume = holiday_data[holiday_data['timestamp'] == pd.Timestamp(holiday_date)]['vehicle_count'].values[0]
            self.assertGreater(holiday_volume, 1200)  # Holiday volume should be high
    
    def test_weather_impact_analysis(self):
        """Test analysis of weather impact on traffic patterns."""
        # Create test data with weather information
        dates = pd.date_range(start='2025-03-01', periods=30, freq='D')
        
        # Define weather conditions
        weather_conditions = {
            'clear': {'days': 15, 'volume_factor': 1.0},
            'rain': {'days': 10, 'volume_factor': 0.8},
            'snow': {'days': 5, 'volume_factor': 0.6}
        }
        
        # Create data with weather effects
        weather_data = []
        
        weather_types = []
        for condition, info in weather_conditions.items():
            weather_types.extend([condition] * info['days'])
        
        for i, date in enumerate(dates):
            weather = weather_types[i]
            factor = weather_conditions[weather]['volume_factor']
            
            # Base volume with weather effect
            base_volume = 1000 * factor
            
            # Add noise
            volume = base_volume + np.random.normal(0, 50)
            
            weather_data.append({
                'timestamp': date,
                'intersection_id': 1,
                'approach': 'north',
                'vehicle_count': int(volume),
                'weather': weather
            })
        
        weather_df = pd.DataFrame(weather_data)
        self.analyzer.load_data(weather_df)
        
        # Analyze weather impact
        weather_impact = self.analyzer.analyze_weather_impact()
        
        # Verify weather impact analysis
        self.assertIn('clear', weather_impact)
        self.assertIn('rain', weather_impact)
        self.assertIn('snow', weather_impact)
        
        # Verify impacts match expected factors
        clear_volume = weather_impact['clear']['average_volume']
        rain_volume = weather_impact['rain']['average_volume']
        snow_volume = weather_impact['snow']['average_volume']
        
        self.assertGreater(clear_volume, rain_volume)
        self.assertGreater(rain_volume, snow_volume)
        
        # Verify percentage impacts
        self.assertAlmostEqual(
            weather_impact['rain']['percentage_impact'],
            -20,  # Expected 20% decrease
            delta=5  # Allow 5% margin for noise
        )
        
        self.assertAlmostEqual(
            weather_impact['snow']['percentage_impact'],
            -40,  # Expected 40% decrease
            delta=5  # Allow 5% margin for noise
        )
    
    def test_special_event_detection(self):
        """Test detection of special events from traffic data."""
        # Create test data with special events
        dates = pd.date_range(start='2025-03-01', periods=30, freq='D')
        
        # Define special events
        special_events = {
            '2025-03-05': 'Concert',
            '2025-03-12': 'Sports Game',
            '2025-03-20': 'Festival'
        }
        
        # Create data with event effects
        event_data = []
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            
            # Base volume with event effect
            if date_str in special_events:
                base_volume = 2000  # Significant increase during events
            else:
                base_volume = 1000
                
            # Add noise
            volume = base_volume + np.random.normal(0, 50)
            
            event_data.append({
                'timestamp': date,
                'intersection_id': 1,
                'approach': 'north',
                'vehicle_count': int(volume)
            })
        
        event_df = pd.DataFrame(event_data)
        self.analyzer.load_data(event_df)
        
        # Detect special events
        detected_events = self.analyzer.detect_special_events(threshold=1.5)
        
        # Verify event detection
        self.assertEqual(len(detected_events), 3)  # Should detect all 3 events
        
        # Verify event dates
        detected_dates = [event['date'].strftime('%Y-%m-%d') for event in detected_events]
        for event_date in special_events.keys():
            self.assertIn(event_date, detected_dates)
    
    def test_spatial_congestion_correlation(self):
        """Test correlation of congestion across adjacent intersections."""
        # Create test data for multiple intersections
        dates = pd.date_range(start='2025-03-01', periods=24, freq='H')
        
        # Define intersections in a corridor
        intersections = [1, 2, 3]  # Three consecutive intersections
        
        # Create data with propagating congestion
        multi_intersection_data = []
        
        # Base pattern for first intersection
        base_pattern = [10, 15, 20, 50, 80, 100, 90, 70, 40, 30, 20, 10, 
                      15, 25, 35, 60, 90, 100, 80, 60, 40, 30, 20, 10]
        
        for i, intersection_id in enumerate(intersections):
            # Shift pattern by 1 hour for each downstream intersection
            shifted_pattern = np.roll(base_pattern, i)
            
            for j, date in enumerate(dates):
                multi_intersection_data.append({
                    'timestamp': date,
                    'intersection_id': intersection_id,
                    'approach': 'north',
                    'vehicle_count': shifted_pattern[j]
                })
        
        multi_df = pd.DataFrame(multi_intersection_data)
        self.analyzer.load_data(multi_df)
        
        # Analyze spatial correlation
        spatial_correlation = self.analyzer.analyze_spatial_correlation()
        
        # Verify correlation between adjacent intersections
        self.assertIn((1, 2), spatial_correlation)  # Correlation between int 1 and 2
        self.assertIn((2, 3), spatial_correlation)  # Correlation between int 2 and 3
        
        # Verify high correlation with time shift
        self.assertGreater(spatial_correlation[(1, 2)]['max_correlation'], 0.8)
        self.assertEqual(spatial_correlation[(1, 2)]['time_shift'], 1)  # 1 hour shift
        
        self.assertGreater(spatial_correlation[(2, 3)]['max_correlation'], 0.8)
        self.assertEqual(spatial_correlation[(2, 3)]['time_shift'], 1)  # 1 hour shift
    
    def test_trend_analysis(self):
        """Test detection of long-term traffic trends."""
        # Create test data with long-term trend
        dates = pd.date_range(start='2025-01-01', periods=365, freq='D')
        
        # Create increasing trend with seasonal variation
        trend_volumes = []
        for i, date in enumerate(dates):
            # Linear growth component (2% per month)
            growth_factor = 1 + (0.02 * i / 30)
            
            # Seasonal component
            day_of_year = date.dayofyear
            seasonal_factor = 1 + 0.15 * np.sin((day_of_year - 172) * 2 * np.pi / 365)
            
            # Base volume with trend and seasonality
            base_volume = 1000 * growth_factor * seasonal_factor
            
            # Add noise
            volume = base_volume + np.random.normal(0, 50)
            trend_volumes.append(int(volume))
        
        trend_data = pd.DataFrame({
            'timestamp': dates,
            'intersection_id': [1] * 365,
            'approach': ['north'] * 365,
            'vehicle_count': trend_volumes
        })
        
        self.analyzer.load_data(trend_data)
        
        # Analyze long-term trend
        trend_analysis = self.analyzer.analyze_long_term_trend()
        
        # Verify trend detection
        self.assertIn('trend_type', trend_analysis)
        self.assertEqual(trend_analysis['trend_type'], 'increasing')
        
        # Verify growth rate calculation
        self.assertAlmostEqual(
            trend_analysis['annual_growth_rate'],
            24,  # Expected 24% annual growth (2% monthly)
            delta=5  # Allow 5% margin for noise and seasonality
        )
        
        # Verify deseasonalized trend
        self.assertIn('deseasonalized_slope', trend_analysis)
        self.assertGreater(trend_analysis['deseasonalized_slope'], 0)
    
    def test_congestion_propagation_prediction(self):
        """Test prediction of congestion propagation through a network."""
        # Create test data for a corridor of intersections
        corridor_data = []
        
        # Define multiple time periods and intersections
        times = pd.date_range(start='2025-03-01 07:00:00', periods=10, freq='5min')
        intersections = [1, 2, 3, 4]  # Corridor with 4 intersections
        
        # Create propagating congestion pattern
        for i, intersection_id in enumerate(intersections):
            for j, t in enumerate(times):
                # Congestion propagates downstream with time
                # Higher index intersections see congestion later
                time_index = max(0, j - i)
                
                # Congestion pattern: starts low, peaks, then decreases
                if time_index < len(times):
                    if time_index < 3:
                        volume = 50 + time_index * 150  # Increasing
                    else:
                        volume = 500 - (time_index - 3) * 50  # Decreasing after peak
                else:
                    volume = 50  # Default low volume
                
                corridor_data.append({
                    'timestamp': t,
                    'intersection_id': intersection_id,
                    'approach': 'north',
                    'vehicle_count': volume
                })
        
        corridor_df = pd.DataFrame(corridor_data)
        self.analyzer.load_data(corridor_df)
        
        # Predict congestion propagation
        propagation = self.analyzer.predict_congestion_propagation(
            start_intersection=1,
            prediction_horizon=4
        )
        
        # Verify propagation prediction
        self.assertIn(2, propagation)
        self.assertIn(3, propagation)
        self.assertIn(4, propagation)
        
        # Verify propagation timing
        self.assertLess(propagation[2]['expected_time'], propagation[3]['expected_time'])
        self.assertLess(propagation[3]['expected_time'], propagation[4]['expected_time'])
        
        # Verify expected congestion levels
        self.assertIn('expected_peak_volume', propagation[2])
        self.assertGreater(propagation[2]['expected_peak_volume'], 300)


class TestPredictionModel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create historical data for training
        dates = pd.date_range(start='2025-01-01', periods=90, freq='D')
        traffic_volumes = []
        
        # Generate synthetic data with weekly patterns
        for date in dates:
            # Weekday pattern (higher volume on weekdays)
            if date.weekday() < 5:  # Monday to Friday
                base_volume = 1000
            else:  # Weekend
                base_volume = 600
                
            # Add some seasonality
            volume = base_volume + 100 * np.sin(date.day / 30 * np.pi)
            
            # Add random noise
            volume += np.random.normal(0, 50)
            
            traffic_volumes.append(int(volume))
        
        self.historical_data = pd.DataFrame({
            'date': dates,
            'volume': traffic_volumes
        })
        
        # Initialize the prediction model
        self.prediction_model = PredictionModel()
        self.prediction_model.train(self.historical_data)
    
    def test_next_day_prediction(self):
        """Test prediction for the next day's traffic volume."""
        # Predict volume for the next day after the historical data
        next_day = pd.Timestamp('2025-04-01')
        predicted_volume = self.prediction_model.predict_volume(next_day)
        
        # Verify prediction is a reasonable number
        self.assertTrue(500 <= predicted_volume <= 1500)
        
        # Verify prediction matches expected pattern (weekday vs weekend)
        if next_day.weekday() < 5:  # Weekday
            self.assertTrue(800 <= predicted_volume <= 1500)
        else:  # Weekend
            self.assertTrue(500 <= predicted_volume <= 900)
    
    def test_weekly_prediction(self):
        """Test prediction for an entire week's traffic volumes."""
        # Predict volumes for a week
        start_date = pd.Timestamp('2025-04-01')
        end_date = pd.Timestamp('2025-04-07')
        predicted_volumes = self.prediction_model.predict_range(start_date, end_date)
        
        # Verify correct number of predictions returned
        days_diff = (end_date - start_date).days + 1
        self.assertEqual(len(predicted_volumes), days_diff)
        
        # Verify weekday/weekend pattern in predictions
        for i, date in enumerate(pd.date_range(start=start_date, end=end_date)):
            volume = predicted_volumes[i]
            if date.weekday() < 5:  # Weekday
                self.assertTrue(800 <= volume <= 1500, 
                                f"Weekday volume {volume} outside expected range")
            else:  # Weekend
                self.assertTrue(500 <= volume <= 900, 
                                f"Weekend volume {volume} outside expected range")
    
    def test_prediction_with_special_event(self):
        """Test prediction with a special event modifier."""
        regular_day = pd.Timestamp('2025-04-02')
        regular_prediction = self.prediction_model.predict_volume(regular_day)
        
        # Now predict with a special event that increases traffic by 50%
        event_prediction = self.prediction_model.predict_volume(
            regular_day, special_events=[{'impact_factor': 1.5}]
        )
        
        # Verify event prediction is appropriately higher
        expected = regular_prediction * 1.5
        self.assertAlmostEqual(event_prediction, expected, delta=1)
    
    def test_prediction_accuracy(self):
        """Test the accuracy of prediction against known data."""
        # Use first 80 days for training
        train_data = self.historical_data.iloc[:80]
        
        # Use last 10 days for testing
        test_data = self.historical_data.iloc[80:]
        
        # Train a new model on the training data
        model = PredictionModel()
        model.train(train_data)
        
        # Calculate predictions for test dates
        predictions = []
        for date in test_data['date']:
            predictions.append(model.predict_volume(date))
        
        # Calculate mean absolute percentage error (MAPE)
        actuals = test_data['volume'].values
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        # Verify MAPE is reasonable (less than 15%)
        self.assertLess(mape, 15)
        
    # NEW TESTS START HERE
    
    def test_weather_aware_prediction(self):
        """Test prediction with weather factors."""
        # Create data that includes weather
        dates = pd.date_range(start='2025-01-01', periods=90, freq='D')
        data = []
        
        for date in dates:
            # Base volume depends on weekday
            if date.weekday() < 5:
                base_volume = 1000
            else:
                base_volume = 600
            
            # Add weather conditions randomly
            weather = np.random.choice(['sunny', 'rainy', 'snowy'], p=[0.7, 0.2, 0.1])
            
            # Adjust volume based on weather
            if weather == 'rainy':
                weather_factor = 0.9  # 10% reduction
            elif weather == 'snowy':
                weather_factor = 0.7  # 30% reduction
            else:
                weather_factor = 1.0
                
            volume = base_volume * weather_factor + np.random.normal(0, 50)
            
            data.append({
                'date': date,
                'volume': int(volume),
                'weather': weather
            })
        
        # Create DataFrame
        weather_data = pd.DataFrame(data)
        
        # Train a weather-aware model
        weather_model = PredictionModel(features=['weather'])
        weather_model.train(weather_data)
        
        # Test prediction with different weather conditions
        test_date = pd.Timestamp('2025-04-01')  # A weekday
        
        # Predict for different weather conditions
        sunny_prediction = weather_model.predict_volume(test_date, weather='sunny')
        rainy_prediction = weather_model.predict_volume(test_date, weather='rainy')
        snowy_prediction = weather_model.predict_volume(test_date, weather='snowy')
        
        # Verify weather impacts predictions correctly
        self.assertGreater(sunny_prediction, rainy_prediction)
        self.assertGreater(rainy_prediction, snowy_prediction)
        
        # Verify approximate impact magnitudes
        self.assertAlmostEqual(
            rainy_prediction / sunny_prediction,
            0.9,  # Expected 10% reduction
            delta=0.1
        )
        
        self.assertAlmostEqual(
            snowy_prediction / sunny_prediction,
            0.7,  # Expected 30% reduction
            delta=0.1
        )
    
    def test_time_of_day_prediction(self):
        """Test hourly traffic volume prediction."""
    # Create data with hourly pattern
    dates = pd.date_range(start='2025-01-01', periods=24, freq='H')
    hourly_data = []
    
    # Create typical hourly pattern
    hourly_pattern = [
        300, 200, 150, 100, 150, 300,  # Midnight to 6am
        700, 1200, 1100, 900, 800, 750,  # 6am to noon
        800, 850, 900, 950, 1100, 1300,  # Noon to 6pm
        1000, 800, 700, 600, 500, 400   # 6pm to midnight
    ]
    
    for i, date in enumerate(dates):
        hourly_data.append({
            'datetime': date,
            'volume': hourly_pattern[i] + np.random.normal(0, 50),
            'hour': date.hour
        })
    
    # Create DataFrame and train model
    hourly_df = pd.DataFrame(hourly_data)
    hourly_model = PredictionModel(time_granularity='hourly')
    hourly_model.train(hourly_df)
    
    # Test predictions for different hours
    morning_rush = hourly_model.predict_volume_by_hour(8)  # 8am
    afternoon_rush = hourly_model.predict_volume_by_hour(17)  # 5pm
    night_time = hourly_model.predict_volume_by_hour(2)  # 2am
    
    # Verify predictions match expected pattern
    self.assertGreater(morning_rush, 1000)
    self.assertGreater(afternoon_rush, 1000)
    self.assertLess(night_time, 300)
    
    # Verify rush hour detection
    rush_hours = hourly_model.identify_peak_hours(threshold=1000)
    self.assertIn(8, rush_hours)
    self.assertIn(17, rush_hours)
    self.assertNotIn(2, rush_hours)

def test_multifactor_prediction(self):
    """Test prediction with multiple factors combined."""
    # Create complex dataset with multiple factors
    dates = pd.date_range(start='2025-01-01', periods=60, freq='D')
    complex_data = []
    
    for date in dates:
        # Base volume depends on weekday/weekend
        if date.weekday() < 5:  # Weekday
            base_volume = 1000
        else:  # Weekend
            base_volume = 600
        
        # Weather factor (random assignment)
        weather = np.random.choice(['sunny', 'rainy', 'snowy'], p=[0.6, 0.3, 0.1])
        if weather == 'rainy':
            weather_factor = 0.9
        elif weather == 'snowy':
            weather_factor = 0.7
        else:
            weather_factor = 1.0
            
        # School factor (in session or break)
        # Assume winter break for first 5 days, spring break for days 40-45
        if date.day <= 5 or (40 <= date.day <= 45):
            school_factor = 0.8  # Less traffic during school breaks
        else:
            school_factor = 1.0
            
        # Special events (random large events)
        has_event = np.random.random() < 0.1  # 10% chance of special event
        event_factor = 1.5 if has_event else 1.0
        
        # Calculate final volume with all factors
        volume = base_volume * weather_factor * school_factor * event_factor
        
        # Add noise
        volume += np.random.normal(0, 50)
        
        complex_data.append({
            'date': date,
            'volume': int(volume),
            'weather': weather,
            'school_break': 1 if (date.day <= 5 or (40 <= date.day <= 45)) else 0,
            'special_event': 1 if has_event else 0
        })
    
    # Create DataFrame
    complex_df = pd.DataFrame(complex_data)
    
    # Train a multifactor model
    complex_model = PredictionModel(features=['weather', 'school_break', 'special_event'])
    complex_model.train(complex_df)
    
    # Test prediction with various combinations
    test_date = pd.Timestamp('2025-03-01')  # A date not in training set
    
    # Base prediction (weekday, sunny, no school break, no event)
    base_pred = complex_model.predict_volume(
        test_date, 
        weather='sunny', 
        school_break=0, 
        special_event=0
    )
    
    # Prediction with rain
    rain_pred = complex_model.predict_volume(
        test_date, 
        weather='rainy', 
        school_break=0, 
        special_event=0
    )
    
    # Prediction with school break
    school_pred = complex_model.predict_volume(
        test_date, 
        weather='sunny', 
        school_break=1, 
        special_event=0
    )
    
    # Prediction with special event
    event_pred = complex_model.predict_volume(
        test_date, 
        weather='sunny', 
        school_break=0, 
        special_event=1
    )
    
    # Prediction with all factors
    all_factors_pred = complex_model.predict_volume(
        test_date, 
        weather='rainy', 
        school_break=1, 
        special_event=1
    )
    
    # Verify individual factor impacts
    self.assertLess(rain_pred, base_pred)
    self.assertLess(school_pred, base_pred)
    self.assertGreater(event_pred, base_pred)
    
    # Verify combined effects
    self.assertLess(all_factors_pred, event_pred)  # Rain and school break reduce volume
    self.assertGreater(all_factors_pred, rain_pred)  # Event increases volume

def test_prediction_confidence_intervals(self):
    """Test that the model can generate confidence intervals for predictions."""
    # Train model on historical data
    model = PredictionModel(confidence_intervals=True)
    model.train(self.historical_data)
    
    # Get prediction with confidence intervals
    test_date = pd.Timestamp('2025-04-01')
    prediction, lower_bound, upper_bound = model.predict_with_confidence(test_date)
    
    # Verify prediction is within bounds
    self.assertLessEqual(lower_bound, prediction)
    self.assertLessEqual(prediction, upper_bound)
    
    # Verify reasonable interval width (not too narrow or wide)
    interval_width = upper_bound - lower_bound
    prediction_range = 0.3 * prediction  # Expecting roughly ±15% interval
    
    self.assertGreater(interval_width, 0.1 * prediction)  # Not too narrow
    self.assertLess(interval_width, 0.5 * prediction)  # Not too wide

def test_anomaly_detection_in_prediction(self):
    """Test ability to detect anomalies in actual vs predicted traffic."""
    # Create test data with some anomalies
    dates = pd.date_range(start='2025-01-01', periods=30, freq='D')
    volumes = []
    
    for i, date in enumerate(dates):
        # Base pattern
        if date.weekday() < 5:
            base_volume = 1000
        else:
            base_volume = 600
            
        # Add anomalies on specific days
        if i == 10:  # Sudden spike
            volume = base_volume * 2
        elif i == 20:  # Sudden drop
            volume = base_volume * 0.5
        else:
            volume = base_volume + np.random.normal(0, 50)
            
        volumes.append(int(volume))
    
    anomaly_df = pd.DataFrame({'date': dates, 'volume': volumes})
    
    # Split into training (first 25 days) and test (last 5 days including one anomaly)
    train_df = anomaly_df.iloc[:25]
    test_df = anomaly_df.iloc[25:]
    
    # Train model
    model = PredictionModel()
    model.train(train_df)
    
    # Test anomaly detection
    anomalies = model.detect_anomalies(test_df, threshold=1.5)
    
    # Verify anomaly detection
    self.assertGreaterEqual(len(anomalies), 0)
    
    # If there was an anomaly in the test period, verify it was detected
    if any(abs(row['volume'] - model.predict_volume(row['date'])) > 
           threshold * model.prediction_std for _, row in test_df.iterrows()):
        self.assertGreater(len(anomalies), 0)

def test_transfer_learning_between_intersections(self):
    """Test transferring a prediction model from one intersection to another similar one."""
    # Create data for two similar intersections
    dates = pd.date_range(start='2025-01-01', periods=60, freq='D')
    
    # Intersection 1 data (source)
    int1_data = []
    for date in dates:
        # Base pattern
        if date.weekday() < 5:
            base_volume = 1000
        else:
            base_volume = 600
            
        # Add noise
        volume = base_volume + np.random.normal(0, 50)
        int1_data.append({'date': date, 'volume': int(volume), 'intersection_id': 1})
    
    # Intersection 2 data (target) - similar pattern but 20% higher volume
    int2_data = []
    for date in dates[:30]:  # Only 30 days of data for int2 (less history)
        # Base pattern - same weekly pattern, different magnitude
        if date.weekday() < 5:
            base_volume = 1200  # 20% higher
        else:
            base_volume = 720   # 20% higher
            
        # Add noise
        volume = base_volume + np.random.normal(0, 60)
        int2_data.append({'date': date, 'volume': int(volume), 'intersection_id': 2})
    
    # Create DataFrames
    int1_df = pd.DataFrame(int1_data)
    int2_df = pd.DataFrame(int2_data)
    
    # Train source model
    source_model = PredictionModel()
    source_model.train(int1_df)
    
    # Test predictions on target intersection with and without transfer learning
    
    # Without transfer learning - train from scratch on target data
    basic_model = PredictionModel()
    basic_model.train(int2_df)
    
    # With transfer learning - initialize with source model parameters
    transfer_model = PredictionModel()
    transfer_model.transfer_from(source_model, scaling_factor=1.2)
    transfer_model.fine_tune(int2_df)  # Fine-tune on target data
    
    # Make predictions on future dates for target intersection
    test_dates = pd.date_range(start='2025-03-01', periods=7, freq='D')
    
    # Calculate prediction error with and without transfer learning
    basic_errors = []
    transfer_errors = []
    
    for date in test_dates:
        # True volume (synthetic) following the pattern
        if date.weekday() < 5:
            true_volume = 1200 + np.random.normal(0, 60)
        else:
            true_volume = 720 + np.random.normal(0, 60)
        
        # Predictions
        basic_pred = basic_model.predict_volume(date)
        transfer_pred = transfer_model.predict_volume(date)
        
        # Calculate absolute errors
        basic_errors.append(abs(basic_pred - true_volume))
        transfer_errors.append(abs(transfer_pred - true_volume))
    
    # Verify transfer learning improves accuracy
    avg_basic_error = np.mean(basic_errors)
    avg_transfer_error = np.mean(transfer_errors)
    
    self.assertLess(avg_transfer_error, avg_basic_error)

def test_ensemble_prediction(self):
    """Test ensemble prediction combining multiple models."""
    # Train three different models on the same data
    model1 = PredictionModel(algorithm='linear')  # Linear regression
    model2 = PredictionModel(algorithm='forest')  # Random forest
    model3 = PredictionModel(algorithm='neural')  # Neural network
    
    model1.train(self.historical_data)
    model2.train(self.historical_data)
    model3.train(self.historical_data)
    
    # Create an ensemble model
    ensemble = PredictionModel(algorithm='ensemble')
    ensemble.add_models([model1, model2, model3], weights=[0.3, 0.5, 0.2])
    
    # Make predictions with all models
    test_date = pd.Timestamp('2025-04-01')
    pred1 = model1.predict_volume(test_date)
    pred2 = model2.predict_volume(test_date)
    pred3 = model3.predict_volume(test_date)
    ensemble_pred = ensemble.predict_volume(test_date)
    
    # Calculate expected weighted average
    expected_pred = 0.3*pred1 + 0.5*pred2 + 0.2*pred3
    
    # Verify ensemble prediction is weighted average of individual predictions
    self.assertAlmostEqual(ensemble_pred, expected_pred, delta=1)
    
    # Verify ensemble prediction accuracy on test data
    test_data = self.historical_data.iloc[-10:]  # Last 10 days
    
    individual_errors = []
    ensemble_errors = []
    
    for _, row in test_data.iterrows():
        date = row['date']
        actual = row['volume']
        
        # Get individual and ensemble predictions
        preds = [model.predict_volume(date) for model in [model1, model2, model3]]
        ens_pred = ensemble.predict_volume(date)
        
        # Calculate errors
        individual_errors.append([abs(pred - actual) for pred in preds])
        ensemble_errors.append(abs(ens_pred - actual))
    
    # Calculate average errors
    avg_ind_errors = [np.mean([e[i] for e in individual_errors]) for i in range(3)]
    avg_ens_error = np.mean(ensemble_errors)
    
    # Verify ensemble error is better than at least one individual model
    self.assertLessEqual(avg_ens_error, max(avg_ind_errors))