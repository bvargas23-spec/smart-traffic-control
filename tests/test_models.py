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
        dates = pd.date_range(start='2025-