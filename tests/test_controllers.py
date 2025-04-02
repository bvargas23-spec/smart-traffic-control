import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest # type: ignore
import time
import sys
import os
import threading

# Assuming your controllers are in a module called "controllers"
# We'll use patch to mock them during testing
try:
    from controllers import SignalController, SensorController, ControllerStatus # type: ignore
except ImportError:
    # Create dummy classes for testing purposes if the real module isn't available
    class SignalController:
        def __init__(self, hardware_interface=None):
            self.hardware_interface = hardware_interface
            self.status = ControllerStatus.NORMAL
            self.signals = []
            self.phases = []
            self.intersection_id = None
            self.sync_partners = []
            self.sync_clock_offset = 0
    
    class SensorController:
        def __init__(self, sensor_interface=None):
            self.sensor_interface = sensor_interface
            self.fault_log = {}
    
    class ControllerStatus:
        NORMAL = "NORMAL"
        ERROR = "ERROR"
        MAINTENANCE = "MAINTENANCE"
        EMERGENCY = "EMERGENCY"


class TestSignalController(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock hardware interface
        self.mock_hardware = Mock()
        
        # Configure the mock hardware to return specific values
        self.mock_hardware.get_status.return_value = "OK"
        self.mock_hardware.send_command.return_value = True
        
        # Create our controller with the mock hardware
        self.controller = SignalController(hardware_interface=self.mock_hardware)
    
    def tearDown(self):
        """Clean up after each test method."""
        # Any cleanup steps needed
        pass
    
    def test_initialization(self):
        """Test that the controller initializes correctly."""
        self.assertEqual(self.controller.status, ControllerStatus.NORMAL)
        self.mock_hardware.initialize.assert_called_once()
    
    def test_set_signal_state(self):
        """Test setting a signal to a specific state."""
        # Test changing a signal's state
        result = self.controller.set_signal_state(signal_id=1, state="GREEN")
        
        # Verify the controller called the hardware with correct parameters
        self.mock_hardware.send_command.assert_called_with(signal_id=1, command="SET_STATE", value="GREEN")
        self.assertTrue(result)
    
    def test_emergency_mode(self):
        """Test activating emergency mode."""
        self.controller.activate_emergency_mode()
        
        # Verify all signals were set to the appropriate state for emergency
        self.mock_hardware.send_command.assert_called_with(command="EMERGENCY_MODE")
        self.assertEqual(self.controller.status, ControllerStatus.EMERGENCY)
    
    def test_fault_detection(self):
        """Test the controller detects faults in the hardware."""
        # Configure mock to simulate a hardware fault
        self.mock_hardware.get_status.return_value = "FAULT"
        
        # Check status should detect the fault
        status = self.controller.check_status()
        
        self.assertEqual(status, ControllerStatus.ERROR)
        self.mock_hardware.get_status.assert_called()
    
    def test_phase_transition(self):
        """Test transitioning between signal phases."""
        # Define a simple phase sequence
        phase_sequence = [
            {"signal_id": 1, "state": "GREEN"},
            {"signal_id": 2, "state": "RED"},
            {"signal_id": 3, "state": "RED"}
        ]
        
        # Execute the phase transition
        self.controller.execute_phase(phase_sequence)
        
        # Verify each signal was set correctly
        expected_calls = [
            unittest.mock.call(signal_id=1, command="SET_STATE", value="GREEN"),
            unittest.mock.call(signal_id=2, command="SET_STATE", value="RED"),
            unittest.mock.call(signal_id=3, command="SET_STATE", value="RED")
        ]
        
        self.mock_hardware.send_command.assert_has_calls(expected_calls, any_order=False)
        
    # NEW TESTS START HERE
    
    def test_high_frequency_commands(self):
        """Test controller behavior under rapid command sequences."""
        # Track response times
        response_times = []
        
        # Send 100 rapid commands
        for i in range(100):
            start_time = time.time()
            result = self.controller.set_signal_state(signal_id=1, state="GREEN")
            response_time = time.time() - start_time
            response_times.append(response_time)
            
        # Verify all commands succeeded
        self.assertTrue(all(response_times))
        
        # Check that response time didn't degrade significantly
        self.assertLess(max(response_times), 2 * min(response_times))
    
    def test_concurrent_commands(self):
        """Test controller behavior with concurrent command execution."""
        # Define a test function that sends a command and tracks success
        results = []
        
        def send_command(signal_id, state):
            result = self.controller.set_signal_state(signal_id=signal_id, state=state)
            results.append(result)
        
        # Create threads for concurrent execution
        threads = []
        for i in range(10):
            # Create commands for different signals
            t = threading.Thread(
                target=send_command, 
                args=(i, "GREEN" if i % 2 == 0 else "RED")
            )
            threads.append(t)
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify all commands were successful
        self.assertTrue(all(results))
        
        # Check that correct number of commands were executed
        self.assertEqual(len(results), 10)
    
    def test_configuration_loading(self):
        """Test loading signal configurations."""
        # Create a sample configuration
        config = {
            "intersection_id": "INT-001",
            "signals": [
                {"id": 1, "type": "vehicle", "default_state": "RED"},
                {"id": 2, "type": "vehicle", "default_state": "RED"},
                {"id": 3, "type": "pedestrian", "default_state": "DONT_WALK"}
            ],
            "phases": [
                {"id": 1, "signals": [{"id": 1, "state": "GREEN"}, {"id": 3, "state": "DONT_WALK"}]},
                {"id": 2, "signals": [{"id": 2, "state": "GREEN"}, {"id": 3, "state": "WALK"}]}
            ]
        }
        
        # Load the configuration
        result = self.controller.load_configuration(config)
        
        # Verify configuration was loaded
        self.assertTrue(result)
        self.assertEqual(self.controller.intersection_id, "INT-001")
        self.assertEqual(len(self.controller.signals), 3)
        self.assertEqual(len(self.controller.phases), 2)
    
    def test_configuration_update(self):
        """Test updating configuration while system is running."""
        # Start with initial configuration
        initial_config = {
            "intersection_id": "INT-001",
            "signals": [
                {"id": 1, "type": "vehicle", "default_state": "RED"}
            ]
        }
        self.controller.load_configuration(initial_config)
        
        # Now update while system is running
        self.controller.status = ControllerStatus.NORMAL  # Simulate running system
        
        update_config = {
            "intersection_id": "INT-001",
            "signals": [
                {"id": 1, "type": "vehicle", "default_state": "RED"},
                {"id": 2, "type": "vehicle", "default_state": "RED"}  # Added signal
            ]
        }
        
        result = self.controller.update_configuration(update_config)
        
        # Verify configuration was updated
        self.assertTrue(result)
        self.assertEqual(len(self.controller.signals), 2)
        
        # Verify system remains in NORMAL state
        self.assertEqual(self.controller.status, ControllerStatus.NORMAL)
    
    def test_communication_timeout(self):
        """Test controller behavior when hardware communication times out."""
        # Configure mock to simulate timeout
        self.mock_hardware.send_command.side_effect = TimeoutError("Communication timeout")
        
        # Attempt to set signal state
        result = self.controller.set_signal_state(signal_id=1, state="GREEN")
        
        # Verify command failed
        self.assertFalse(result)
        
        # Verify controller status was updated
        self.assertEqual(self.controller.status, ControllerStatus.ERROR)
        
        # Verify fallback action was taken
        self.mock_hardware.enter_failsafe.assert_called_once()
    
    def test_partial_failure_recovery(self):
        """Test recovery from partial system failure."""
        # Mock hardware to simulate partial failure
        self.mock_hardware.get_signal_status.return_value = {
            1: "OK",
            2: "FAULT",
            3: "OK"
        }
        
        # Attempt recovery
        recovery_result = self.controller.recover_partial_failure()
        
        # Verify controller attempted to reset only the faulty signal
        self.mock_hardware.reset_signal.assert_called_with(signal_id=2)
        
        # Verify controller remains in operational state
        self.assertEqual(self.controller.status, ControllerStatus.NORMAL)
    
    def test_controller_synchronization(self):
        """Test synchronization between multiple controllers."""
        # Create a second controller
        second_controller = SignalController(hardware_interface=Mock())
        
        # Set up a synchronization link
        self.controller.add_sync_partner(second_controller)
        
        # Execute a synchronized phase change
        self.controller.execute_synchronized_phase(phase_id=1)
        
        # Verify both controllers executed the phase
        self.mock_hardware.send_command.assert_called()
        second_controller.hardware_interface.send_command.assert_called()
        
        # Verify timing was coordinated (commands sent within 100ms)
        time_diff = abs(
            self.mock_hardware.send_command.call_args[1].get('timestamp', 0) -
            second_controller.hardware_interface.send_command.call_args[1].get('timestamp', 0)
        )
        self.assertLess(time_diff, 0.1)  # Less than 100ms difference
    
    def test_clock_drift_compensation(self):
        """Test compensation for clock drift between controllers."""
        # Create a second controller with simulated drift
        second_controller = SignalController(hardware_interface=Mock())
        
        # Set up drift (500ms ahead)
        self.controller.sync_clock_offset = 0.5
        
        # Add as sync partner
        self.controller.add_sync_partner(second_controller)
        
        # Execute synchronized command
        self.controller.execute_synchronized_command("SET_STATE", value="GREEN")
        
        # Verify timestamp was adjusted for drift
        main_timestamp = self.mock_hardware.send_command.call_args[1].get('timestamp')
        partner_timestamp = second_controller.hardware_interface.send_command.call_args[1].get('timestamp')
        
        # The difference should be approximately equal to the configured offset
        self.assertAlmostEqual(partner_timestamp - main_timestamp, 0.5, delta=0.05)


class TestSensorController(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock sensor interface
        self.mock_sensors = Mock()
        
        # Configure the mock to return specific sensor readings
        self.mock_sensors.read_vehicle_count.return_value = 10
        self.mock_sensors.read_vehicle_speed.return_value = 35.5
        self.mock_sensors.read_occupancy.return_value = 0.45
        
        # Create our sensor controller with the mock
        self.sensor_controller = SensorController(sensor_interface=self.mock_sensors)
    
    def test_get_vehicle_count(self):
        """Test getting vehicle count from a sensor."""
        count = self.sensor_controller.get_vehicle_count(sensor_id=1)
        
        self.assertEqual(count, 10)
        self.mock_sensors.read_vehicle_count.assert_called_with(sensor_id=1)
    
    def test_get_average_speed(self):
        """Test getting the average speed from a sensor."""
        speed = self.sensor_controller.get_average_speed(sensor_id=1)
        
        self.assertEqual(speed, 35.5)
        self.mock_sensors.read_vehicle_speed.assert_called_with(sensor_id=1)
    
    def test_get_lane_occupancy(self):
        """Test getting lane occupancy percentage."""
        occupancy = self.sensor_controller.get_lane_occupancy(sensor_id=1)
        
        self.assertEqual(occupancy, 0.45)
        self.mock_sensors.read_occupancy.assert_called_with(sensor_id=1)
    
    def test_sensor_fault_detection(self):
        """Test handling of sensor faults."""
        # Configure mock to raise an exception to simulate faulty sensor
        self.mock_sensors.read_vehicle_count.side_effect = Exception("Sensor fault")
        
        # The controller should handle this gracefully
        count = self.sensor_controller.get_vehicle_count(sensor_id=1)
        
        # Should return a default value or indicator of error
        self.assertEqual(count, -1)  # Assuming -1 is the error code
        
        # And log the error
        self.assertTrue(self.sensor_controller.has_fault(sensor_id=1))
    
    def test_all_sensors_status(self):
        """Test getting status of all sensors."""
        # Configure mock to return specific statuses for different sensors
        self.mock_sensors.get_all_statuses.return_value = {
            1: "OK",
            2: "OK",
            3: "FAULT"
        }
        
        statuses = self.sensor_controller.get_all_sensor_statuses()
        
        self.assertEqual(statuses[1], "OK")
        self.assertEqual(statuses[2], "OK")
        self.assertEqual(statuses[3], "FAULT")
        self.mock_sensors.get_all_statuses.assert_called_once()
        
    # NEW TESTS START HERE
    
    def test_sensor_data_aggregation(self):
        """Test aggregation of data from multiple sensors."""
        # Configure multiple sensor readings
        self.mock_sensors.read_vehicle_count.side_effect = [10, 15, 20]
        
        # Aggregate data from multiple sensors
        total_count = self.sensor_controller.get_total_vehicle_count(sensor_ids=[1, 2, 3])
        
        # Verify correct aggregation
        self.assertEqual(total_count, 45)  # 10 + 15 + 20
        
        # Verify all sensors were queried
        self.assertEqual(self.mock_sensors.read_vehicle_count.call_count, 3)
    
    def test_sensor_data_averaging(self):
        """Test averaging of data from multiple sensors."""
        # Configure multiple sensor readings
        self.mock_sensors.read_vehicle_speed.side_effect = [30, 40, 50]
        
        # Average speed from multiple sensors
        avg_speed = self.sensor_controller.get_corridor_average_speed(sensor_ids=[1, 2, 3])
        
        # Verify correct averaging
        self.assertEqual(avg_speed, 40)  # (30 + 40 + 50) / 3
    
    def test_sensor_data_filtering(self):
        """Test filtering of anomalous sensor data."""
        # Configure mock with some anomalous readings
        self.mock_sensors.read_vehicle_count.side_effect = [10, 100, 15]  # 100 is an outlier
        
        # Get filtered data
        filtered_count = self.sensor_controller.get_filtered_vehicle_count(sensor_ids=[1, 2, 3])
        
        # Verify outlier was filtered out
        self.assertAlmostEqual(filtered_count, 12.5)  # Average of 10 and 15, excluding 100
    
    def test_sensor_calibration(self):
        """Test sensor calibration process."""
        # Configure mock for calibration
        self.mock_sensors.read_raw_value.return_value = 500
        self.mock_sensors.get_calibration_factor.return_value = 0.2
        
        # Get calibrated reading
        calibrated_value = self.sensor_controller.get_calibrated_reading(sensor_id=1)
        
        # Verify calibration was applied
        self.assertEqual(calibrated_value, 100)  # 500 * 0.2
        
        # Verify correct methods were called
        self.mock_sensors.read_raw_value.assert_called_with(sensor_id=1)
        self.mock_sensors.get_calibration_factor.assert_called_with(sensor_id=1)
    
    def test_sensor_data_transformation(self):
        """Test transformation of sensor data to different units."""
        # Configure mock for kph speed
        self.mock_sensors.read_vehicle_speed.return_value = 100  # kph
        
        # Get speed in mph
        mph_speed = self.sensor_controller.get_speed_in_mph(sensor_id=1)
        
        # Verify conversion
        self.assertAlmostEqual(mph_speed, 62.1, delta=0.1)  # 100 kph ≈ 62.1 mph
    
    def test_sensor_high_frequency_sampling(self):
        """Test high-frequency sampling from sensors."""
        # Configure mock for sampling
        sample_values = list(range(100))  # 0-99
        self.mock_sensors.read_value.side_effect = sample_values
        
        # Collect samples
        samples = self.sensor_controller.collect_samples(sensor_id=1, count=100)
        
        # Verify correct number of samples
        self.assertEqual(len(samples), 100)
        
        # Verify sample values
        self.assertEqual(samples, sample_values)
    
    def test_sensor_data_integration(self):
        """Test integration of sensor data over time."""
        # Configure mock for time-series data
        self.mock_sensors.get_time_series.return_value = [10, 20, 30, 40, 50]
        
        # Integrate data (calculate area under curve)
        integrated_value = self.sensor_controller.integrate_sensor_data(sensor_id=1, duration=5)
        
        # Verify integration result
        self.assertEqual(integrated_value, 150)  # 10 + 20 + 30 + 40 + 50
    
    def test_sensor_data_derivative(self):
        """Test calculation of rate of change in sensor data."""
        # Configure mock for time-series data
        self.mock_sensors.get_time_series.return_value = [10, 15, 25, 40, 60]
        
        # Calculate rate of change
        derivatives = self.sensor_controller.calculate_rate_of_change(sensor_id=1, duration=5)
        
        # Verify derivatives
        expected_derivatives = [5, 10, 15, 20]  # Differences between consecutive values
        self.assertEqual(derivatives, expected_derivatives)


class TestControllerIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures for integration tests."""
        # Create mock interfaces
        self.mock_hardware = Mock()
        self.mock_sensors = Mock()
        
        # Configure default returns
        self.mock_sensors.read_vehicle_count.return_value = 15
        self.mock_hardware.send_command.return_value = True
        
        # Create controllers with mocks
        self.signal_controller = SignalController(hardware_interface=self.mock_hardware)
        self.sensor_controller = SensorController(sensor_interface=self.mock_sensors)
        
    def test_adaptive_signal_timing(self):
        """Test adaptive signal timing based on sensor data."""
        # This is an integration test that verifies the signal timing adapts based on sensor data
        
        # First get traffic data
        vehicle_count = self.sensor_controller.get_vehicle_count(sensor_id=1)
        
        # Now adjust signal timing based on the count
        # Assuming we have a function to do this:
        new_timing = self.signal_controller.adjust_timing_for_volume(signal_id=1, volume=vehicle_count)
        
        # Verify hardware command was sent with new timing
        self.mock_hardware.send_command.assert_called_with(
            signal_id=1, 
            command="SET_TIMING", 
            value=new_timing
        )
    
    def test_emergency_vehicle_preemption(self):
        """Test emergency vehicle preemption system."""
        # Configure mock to simulate emergency vehicle detection
        self.mock_sensors.detect_emergency_vehicle.return_value = True
        
        # Should trigger preemption if emergency vehicle detected
        if self.sensor_controller.detect_emergency_vehicle(intersection_id=1):
            self.signal_controller.activate_emergency_mode()
        
        # Verify proper commands were sent to the hardware
        self.mock_hardware.send_command.assert_called_with(command="EMERGENCY_MODE")
        
    def test_fault_recovery(self):
        """Test system recovery after fault detection and resolution."""
        # First simulate a fault
        self.mock_hardware.get_status.return_value = "FAULT"
        
        # Controller should detect the fault
        status = self.signal_controller.check_status()
        self.assertEqual(status, ControllerStatus.ERROR)
        
        # Now simulate fault resolution
        self.mock_hardware.get_status.return_value = "OK"
        self.mock_hardware.reset.return_value = True
        
        # Controller should recover
        self.signal_controller.recover_from_fault()
        
        # Verify recovery process
        self.mock_hardware.reset.assert_called_once()
        self.assertEqual(self.signal_controller.status, ControllerStatus.NORMAL)
        
    # NEW TESTS START HERE
    
    def test_volume_based_adaptive_control(self):
        """Test fully adaptive control based on traffic volumes."""
        # Configure different volumes for different approaches
        approach_volumes = {
            "north": 100,
            "south": 50,
            "east": 200,
            "west": 150
        }
        
        self.mock_sensors.get_approach_volumes.return_value = approach_volumes
        
        # Execute adaptive control cycle
        self.signal_controller.run_adaptive_control_cycle(sensor_controller=self.sensor_controller)
        
        # Verify east-west got longer green (highest volumes)
        east_west_green = self.signal_controller.current_timing_plan["east_west_green"]
        north_south_green = self.signal_controller.current_timing_plan["north_south_green"]
        
        self.assertGreater(east_west_green, north_south_green)
    
    def test_traffic_prediction_integration(self):
        """Test integration with traffic prediction for proactive control."""
        # Configure predicted volumes (increasing trend)
        current_volumes = {"east": 100, "west": 100}
        predicted_volumes = {"east": 150, "west": 120}
        
        self.mock_sensors.get_current_volumes.return_value = current_volumes
        self.mock_sensors.predict_future_volumes.return_value = predicted_volumes
        
        # Run predictive control
        self.signal_controller.run_predictive_control(
            sensor_controller=self.sensor_controller,
            prediction_horizon=5  # 5 minute prediction
        )
        
        # Verify east-west timing was increased proactively
        self.assertGreater(
            self.signal_controller.current_timing_plan["east_west_green"],
            self.signal_controller.previous_timing_plan["east_west_green"]
        )
    
    def test_coordination_with_adjacent_intersections(self):
        """Test coordination with adjacent intersections."""
        # Configure upstream and downstream intersections
        upstream_controller = SignalController(hardware_interface=Mock())
        downstream_controller = SignalController(hardware_interface=Mock())
        
        # Add to coordination group
        self.signal_controller.add_to_coordination_group(
            upstream=upstream_controller,
            downstream=downstream_controller,
            corridor_speed=40  # km/h
        )
        
        # Execute coordinated timing update
        self.signal_controller.update_coordinated_timing()
        
        # Verify offset was adjusted based on travel time
        distance_to_upstream = 500  # meters
        expected_offset = (distance_to_upstream / (40 * 1000/3600))  # seconds
        
        self.assertAlmostEqual(
            self.signal_controller.timing_parameters["offset_to_upstream"],
            expected_offset,
            delta=1  # Allow 1 second difference for calculation precision
        )
    
    def test_pedestrian_demand_responsive_control(self):
        """Test pedestrian responsive signal control."""
        # Configure pedestrian demand
        self.mock_sensors.get_pedestrian_call.return_value = True
        self.mock_sensors.get_pedestrian_waiting_time.return_value = 45  # seconds
        
        # Run pedestrian responsive control
        self.signal_controller.handle_pedestrian_calls(
            sensor_controller=self.sensor_controller,
            max_wait_time=60  # seconds
        )
        
        # Verify pedestrian phase was triggered
        self.mock_hardware.send_command.assert_called_with(
            command="SET_PEDESTRIAN_PHASE",
            value="ACTIVATE"
        )
    
    def test_transit_signal_priority(self):
        """Test transit signal priority integration."""
        # Configure transit vehicle approaching
        self.mock_sensors.detect_transit_vehicle.return_value = {
            "detected": True,
            "distance": 200,  # meters
            "speed": 30,  # km/h
            "line": "Route 42"
        }
        
        # Handle transit priority
        self.signal_controller.handle_transit_priority(
            sensor_controller=self.sensor_controller
        )
        
        # Verify green extension or early green was activated
        priority_calls = [
            call[1].get('command') for call in self.mock_hardware.send_command.call_args_list
            if call[1].get('command') in ['EXTEND_GREEN', 'EARLY_GREEN']
        ]
        
        self.assertGreater(len(priority_calls), 0)
    
    def test_adaptive_recovery_from_oversaturation(self):
        """Test recovery from oversaturated conditions."""
        # Configure oversaturated conditions
        self.mock_sensors.get_saturation_levels.return_value = {
            "north": 1.2,  # Oversaturated (>1.0)
            "south": 1.1,
            "east": 0.8,
            "west": 0.7
        }
        
        # Execute recovery strategy
        self.signal_controller.handle_oversaturation(
            sensor_controller=self.sensor_controller
        )
        
        # Verify north-south (most congested) received priority
        latest_timing = self.signal_controller.current_timing_plan
        self.assertGreater(
            latest_timing["north_south_green"],
            latest_timing["east_west_green"]
        )
        
        # Verify flush mode was activated
        self.mock_hardware.send_command.assert_called_with(
            command="SET_FLUSH_MODE",
            direction="NORTH_SOUTH",
            duration=latest_timing["north_south_green"]
        )
    
    def test_weather_responsive_timing(self):
        """Test weather-responsive signal timing."""
        # Configure adverse weather conditions
        self.mock_sensors.get_weather_conditions.return_value = {
            "condition": "RAIN",
            "intensity": "HEAVY",
            "visibility": "REDUCED"
        }
        
        # Apply weather-responsive timing
        self.signal_controller.adjust_for_weather_conditions(
            sensor_controller=self.sensor_controller
        )
        
        # Verify yellow timing was extended
        default_yellow = self.signal_controller.default_timing_parameters["yellow_time"]
        current_yellow = self.signal_controller.current_timing_plan["yellow_time"]
        
        self.assertGreater(current_yellow, default_yellow)


# Pytest style tests
@pytest.fixture
def mock_hardware():
    """Pytest fixture for hardware interface mock."""
    hardware = Mock()
    hardware.get_status.return_value = "OK"
    hardware.send_command.return_value = True
    hardware.initialize.return_value = True
    return hardware

@pytest.fixture
def signal_controller(mock_hardware):
    """Pytest fixture for initialized signal controller."""
    return SignalController(hardware_interface=mock_hardware)

@pytest.mark.parametrize("signal_id,state,expected_result", [
    (1, "GREEN", True),
    (2, "RED", True),
    (3, "YELLOW", True),
    (4, "FLASHING_RED", True),
])
def test_parametrized_signal_states(signal_controller, mock_hardware, signal_id, state, expected_result):
    """Test setting various signal states using parameterization."""
    result = signal_controller.set_signal_state(signal_id=signal_id, state=state)
    
    mock_hardware.send_command.assert_called_with(signal_id=signal_id, command="SET_STATE", value=state)
    assert result == expected_result

# Mock for hardware that will fail
@pytest.fixture
def failing_hardware():
    """Pytest fixture for hardware that fails commands."""
    hardware = Mock()
    hardware.get_status.return_value = "OK"
    hardware.send_command.return_value = False
    hardware.initialize.return_value = True
    return hardware

@pytest.fixture
def failing_controller(failing_hardware):
    """Controller with hardware that fails commands."""
    return SignalController(hardware_interface=failing_hardware)

def test_hardware_command_failure(failing_controller):
    """Test handling of hardware command failures."""
    result = failing_controller.set_signal_state(signal_id=1, state="GREEN")
    
    # Should return False indicating failure
    assert result is False
    
    # Should update controller status
    assert failing_controller.status == ControllerStatus.ERROR

# NEW TESTS START HERE

@pytest.mark.parametrize("signal_id,state,limit,expected_failures", [
    (1, "GREEN", 1000, 0),  # Should handle 1000 commands
    (2, "YELLOW", 5000, 0),  # Should handle 5000 commands
])
def test_command_volume_limits(signal_controller, mock_hardware, signal_id, state, limit, expected_failures):
    """Test controller handling high volumes of commands."""
    failures = 0
    
    # Send many commands
    for _ in range(limit):
        result = signal_controller.set_signal_state(signal_id=signal_id, state=state)
        if not result:
            failures += 1
    
    # Verify expected number of failures
    assert failures == expected_failures

@pytest.mark.parametrize("boundary_value,expected_result", [
    (0, True),            # Minimum signal ID
    (255, True),          # Maximum signal ID
    (-1, False),          # Below minimum
    (256, False),         # Above maximum
])
def test_boundary_signal_ids(signal_controller, mock_hardware, boundary_value, expected_result):
    """Test handling of boundary signal IDs."""
    result = signal_controller.set_signal_state(signal_id=boundary_value, state="GREEN")
    assert result == expected_result

@pytest.mark.parametrize("invalid_state,expected_result", [
    ("INVALID", False),   # Invalid state
    ("BLUE", False),      # Non-existent state
    ("", False),          # Empty state
    (None, False),        # None state
])
def test_invalid_signal_states(signal_controller, mock_hardware, invalid_state, expected_result):
    """Test handling of invalid signal states."""
    result = signal_controller.set_signal_state(signal_id=1, state=invalid_state)
    assert result == expected_result

@pytest.mark.parametrize("config_file", [
    "valid_config.json",
    "minimal_config.json"
])
def test_configuration_file_loading(signal_controller, config_file, tmp_path):
    """Test loading configurations from files."""
    # Create a temporary config file
    config_path = tmp_path / config_file
    
    # Create valid configuration content
    if config_file == "valid_config.json":
        config = {
            "intersection_id": "TEST-001",
            "signals": [{"id": 1, "type": "vehicle"}]
        }
    else:  # minimal_config.json
        config = {"intersection_id": "TEST-001"}
    
    # Write to file
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    # Test loading
    result = signal_controller.load_configuration_file(str(config_path))
    assert result is True
    assert signal_controller.intersection_id == "TEST-001"

@pytest.mark.parametrize("recovery_scenario", [
    "power_outage",
    "communication_loss",
    "software_reset"
])
def test_recovery_scenarios(signal_controller, mock_hardware, recovery_scenario):
    """Test recovery from different failure scenarios."""
    # Simulate failure based on scenario
    if recovery_scenario == "power_outage":
        signal_controller.status = ControllerStatus.ERROR
        mock_hardware.power_status.return_value = "RESTORED"
    elif recovery_scenario == "communication_loss":
        signal_controller.status = ControllerStatus.ERROR
        mock_hardware.communication_status.return_value = "RESTORED"
    else:  # software_reset
        signal_controller.status = ControllerStatus.ERROR
        mock_hardware.software_status.return_value = "RESTARTED"
    
    # Attempt recovery
    result = signal_controller.recover_from_failure(scenario=recovery_scenario)
    
    # Verify recovery
    assert result is True
    assert signal_controller.status == ControllerStatus.NORMAL

@pytest.fixture
def sensor_system():
    """Fixture for sensor system testing."""
    sensors = Mock()
    sensors.read_vehicle_count.return_value = 10
    return SensorController(sensor_interface=sensors)

@pytest.mark.parametrize("sensor_data,expected_action", [
    ({"congestion_level": "HIGH", "queue_length": 20}, "extend_green"),
    ({"congestion_level": "LOW", "queue_length": 5}, "reduce_green"),
    ({"congestion_level": "MEDIUM", "queue_length": 10}, "maintain")
])
def test_adaptive_response(signal_controller, sensor_system, sensor_data, expected_action):
    """Test adaptive responses to different traffic conditions."""
    # Configure sensor to return specific data
    sensor_system.get_traffic_conditions.return_value = sensor_data
    
    # Get adaptive response
    action = signal_controller.determine_adaptive_action(sensor_system)
    
    # Verify correct action selected
    assert action == expected_action


if __name__ == '__main__':
    unittest.main()    