import unittest
from unittest.mock import Mock, patch
import pytest # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import datetime
import os
import sys

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


class TestSignalTimingOptimizer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample traffic data
        self.traffic_data = {
            'north': {'volume': 800, 'queue_length': 15},
            'south': {'volume': 600, 'queue_length': 10},
            'east': {'volume': 400, 'queue_length': 8},
            'west': {'volume': 300, 'queue_length': 6}
        }
        
        # Create a signal timing optimizer
        self.optimizer = SignalTimingOptimizer()
    
    def test_basic_timing_optimization(self):
        """Test basic signal timing optimization."""
        # Get optimized timing plan
        timing_plan = self.optimizer.optimize(self.traffic_data)
        
        # Verify timing plan structure
        self.assertIn('cycle_length', timing_plan)
        self.assertIn('phases', timing_plan)
        
        # Verify all approaches are included
        approaches = [phase['approach'] for phase in timing_plan['phases']]
        for approach in ['north', 'south', 'east', 'west']:
            self.assertIn(approach, approaches)
        
        # Verify cycle length is reasonable (typically 60-180 seconds)
        self.assertTrue(60 <= timing_plan['cycle_length'] <= 180)
        
        # Verify phase times add up to cycle length
        total_time = sum(phase['duration'] for phase in timing_plan['phases'])
        self.assertEqual(total_time, timing_plan['cycle_length'])
        
        # Verify higher volume approaches get more green time
        north_time = next(phase['duration'] for phase in timing_plan['phases'] 
                         if phase['approach'] == 'north')
        east_time = next(phase['duration'] for phase in timing_plan['phases'] 
                        if phase['approach'] == 'east')
        
        # North has higher volume, so should get more time
        self.assertGreater(north_time, east_time)
    
    def test_timing_with_constraints(self):
        """Test timing optimization with min/max constraints."""
        # Constraint: minimum 15 seconds for each phase
        constraints = {'min_phase_time': 15}
        
        timing_plan = self.optimizer.optimize(self.traffic_data, constraints=constraints)
        
        # Verify minimum constraint is met
        for phase in timing_plan['phases']:
            self.assertGreaterEqual(phase['duration'], constraints['min_phase_time'])
    
    def test_timing_with_pedestrian_crossing(self):
        """Test timing with pedestrian crossing requirements."""
        # Add pedestrian crossing data (typical crossing needs ~15 seconds)
        ped_data = {
            'north': {'ped_count': 10, 'crossing_time': 15},
            'south': {'ped_count': 5, 'crossing_time': 15},
            'east': {'ped_count': 20, 'crossing_time': 15},
            'west': {'ped_count': 15, 'crossing_time': 15}
        }
        
        # Merge with traffic data
        data = self.traffic_data.copy()
        for approach in data:
            data[approach].update(ped_data[approach])
        
        timing_plan = self.optimizer.optimize_with_pedestrians(data)
        
        # Verify east approach (highest ped count) gets sufficient time
        east_time = next(phase['duration'] for phase in timing_plan['phases'] 
                        if phase['approach'] == 'east')
        self.assertGreaterEqual(east_time, ped_data['east']['crossing_time'])


class TestCoordinationModel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample network of intersections
        self.network = {
            1: {'position': (0, 0), 'connections': [2, 3]},
            2: {'position': (500, 0), 'connections': [1, 4]},  # 500m east of 1
            3: {'position': (0, 500), 'connections': [1, 4]},  # 500m north of 1
            4: {'position': (500, 500), 'connections': [2, 3]}  # 500m northeast of 1
        }
        
        # Create sample timing plans for each intersection
        self.timing_plans = {
            1: {'cycle_length': 120, 'offset': 0},
            2: {'cycle_length': 120, 'offset': 0},
            3: {'cycle_length': 120, 'offset': 0},
            4: {'cycle_length': 120, 'offset': 0}
        }
        
        # Create a coordination model
        self.coord_model = CoordinationModel(self.network)
    
    def test_offset_calculation(self):
        """Test calculation of signal offsets for coordination."""
        # Set traffic flow direction from 1 to 2 and 2 to 4
        traffic_flow = {
            (1, 2): 800,  # 800 vehicles from 1 to 2
            (2, 4): 600,  # 600 vehicles from 2 to 4
            (1, 3): 500,  # 500 vehicles from 1 to 3
            (3, 4): 400   # 400 vehicles from 3 to 4
        }
        
        # Average speed of 50 km/h = 13.89 m/s
        avg_speed = 13.89
        
        # Optimize offsets
        coordinated_plan = self.coord_model.optimize_offsets(
            self.timing_plans, traffic_flow, avg_speed
        )
        
        # Verify all intersections have offsets
        for i in self.network:
            self.assertIn('offset', coordinated_plan[i])
        
        # Verify travel time relationship for highest flow path (1 -> 2)
        # If travel time is ~36 seconds (500m / 13.89 m/s), offset should be around that
        travel_time_1_to_2 = 500 / avg_speed
        self.assertAlmostEqual(coordinated_plan[2]['offset'], travel_time_1_to_2, delta=10)
    
    def test_bandwidth_optimization(self):
        """Test green wave bandwidth optimization."""
        # Create arterial path
        arterial = [1, 2, 4]
        
        # Set average speed to 50 km/h
        avg_speed = 13.89  # m/s
        
        # Optimize bandwidth along the arterial
        result = self.coord_model.optimize_bandwidth(arterial, self.timing_plans, avg_speed)
        
        # Verify results contain bandwidth measure
        self.assertIn('bandwidth', result)
        
        # Verify bandwidth is positive and reasonable (typically > 0.3 of cycle)
        self.assertGreater(result['bandwidth'], 0)
        self.assertGreater(result['bandwidth'] / self.timing_plans[1]['cycle_length'], 0.3)
        
        # Verify offsets were adjusted
        for i in arterial[1:]:  # Skip first intersection (reference point)
            self.assertNotEqual(result['offsets'][i], self.timing_plans[i]['offset'])


# Pytest style tests
@pytest.fixture
def sample_traffic_data():
    """Pytest fixture providing sample traffic data."""
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2025-03-01', periods=24, freq='H'),
        'intersection_id': [1] * 24,
        'approach': ['north'] * 24,
        'vehicle_count': [10, 15, 20, 25, 30, 70, 100, 120, 100, 80, 70, 60,
                         50, 60, 70, 90, 110, 130, 120, 90, 60, 40, 20, 10],
        'average_speed': [45, 44, 43, 40, 38, 35, 30, 25, 30, 35, 40, 42,
                         45, 44, 42, 38, 32, 25, 28, 32, 38, 42, 44, 45]
    })

@pytest.fixture
def pattern_analyzer(sample_traffic_data):
    """Pytest fixture for an initialized pattern analyzer."""
    analyzer = TrafficPatternAnalyzer()
    analyzer.load_data(sample_traffic_data)
    return analyzer

@pytest.mark.parametrize("hour,expected_volume", [
    (7, 120),  # 7th hour (peak morning)
    (17, 130)  # 17th hour (peak evening)
])
def test_volume_at_specific_hours(pattern_analyzer, hour, expected_volume):
    """Test getting traffic volumes at specific hours using parameterization."""
    volume = pattern_analyzer.get_volume_at_hour(hour=hour, approach='north')
    assert volume == expected_volume

@pytest.mark.parametrize("threshold,expected_count", [
    (100, 6),  # 6 hours with volume > 100
    (50, 14),  # 14 hours with volume > 50
    (20, 20)   # 20 hours with volume > 20
])
def test_high_volume_hours(pattern_analyzer, threshold, expected_count):
    """Test counting hours with volumes above threshold."""
    high_volume_hours = pattern_analyzer.get_hours_above_threshold(threshold=threshold)
    assert len(high_volume_hours) == expected_count


if __name__ == '__main__':
    unittest.main()