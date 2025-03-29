import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest # type: ignore
import time
import sys
import os

# Assuming your controllers are in a module called "controllers"
# We'll use patch to mock them during testing
try:
    from traffic_control.controllers import SignalController, SensorController, ControllerStatus # type: ignore
except ImportError:
    # Create dummy classes for testing purposes if the real module isn't available
    class SignalController:
        pass
    
    class SensorController:
        pass
    
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


if __name__ == '__main__':
    unittest.main()