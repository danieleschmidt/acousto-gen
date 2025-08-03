"""Test configuration and utilities for Acousto-Gen testing."""

import os
import tempfile
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock
import json


class TestConfig:
    """Centralized test configuration."""
    
    # Test data specifications
    DEFAULT_ARRAY_SIZE = 256
    DEFAULT_FREQUENCY = 40000
    DEFAULT_FIELD_RESOLUTION = (64, 64, 32)
    DEFAULT_FIELD_BOUNDS = {
        "x_min": -0.05, "x_max": 0.05,
        "y_min": -0.05, "y_max": 0.05,
        "z_min": 0.05, "z_max": 0.15
    }
    
    # Performance thresholds
    MAX_OPTIMIZATION_TIME = 60.0  # seconds
    MAX_FIELD_CALCULATION_TIME = 10.0  # seconds
    MAX_API_RESPONSE_TIME = 1.0  # seconds
    
    # Safety limits for testing
    MAX_TEST_PRESSURE = 5000  # Pa
    MAX_TEST_INTENSITY = 15   # W/cm²
    MAX_TEST_TEMPERATURE = 45 # °C
    
    # Test tolerances
    NUMERICAL_TOLERANCE = 1e-6
    PHASE_TOLERANCE = 0.1  # radians
    PRESSURE_TOLERANCE = 50  # Pa
    
    # Test data sizes
    SMALL_FIELD_SIZE = (32, 32, 16)
    MEDIUM_FIELD_SIZE = (64, 64, 32)
    LARGE_FIELD_SIZE = (128, 128, 64)
    
    @classmethod
    def get_test_database_url(cls) -> str:
        """Get temporary database URL for testing."""
        return "sqlite:///:memory:"
    
    @classmethod
    def get_test_data_dir(cls) -> Path:
        """Get test data directory."""
        return Path(__file__).parent / "data"
    
    @classmethod
    def get_fixtures_dir(cls) -> Path:
        """Get fixtures directory."""
        return Path(__file__).parent / "fixtures"


class MockDataGenerator:
    """Generate mock data for testing."""
    
    @staticmethod
    def generate_phases(size: int = 256, seed: Optional[int] = None) -> np.ndarray:
        """Generate random phase array."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.rand(size) * 2 * np.pi
    
    @staticmethod
    def generate_amplitudes(size: int = 256, value: float = 1.0) -> np.ndarray:
        """Generate amplitude array."""
        return np.ones(size) * value
    
    @staticmethod
    def generate_pressure_field(
        shape: tuple = (64, 64, 32),
        max_pressure: float = 3000.0,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Generate realistic pressure field with focus."""
        if seed is not None:
            np.random.seed(seed)
        
        x, y, z = np.meshgrid(
            np.linspace(-0.05, 0.05, shape[0]),
            np.linspace(-0.05, 0.05, shape[1]),
            np.linspace(0.05, 0.15, shape[2]),
            indexing='ij'
        )
        
        # Create focus at center
        focus_x, focus_y, focus_z = 0, 0, 0.1
        sigma = 0.01  # 1cm width
        
        distance = np.sqrt((x - focus_x)**2 + (y - focus_y)**2 + (z - focus_z)**2)
        pressure = max_pressure * np.exp(-(distance / sigma)**2)
        
        return pressure.astype(np.float32)
    
    @staticmethod
    def generate_complex_field(
        shape: tuple = (64, 64, 32),
        max_magnitude: float = 1000.0,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Generate complex pressure field with phase."""
        if seed is not None:
            np.random.seed(seed)
        
        magnitude = np.random.rand(*shape) * max_magnitude
        phase = np.random.rand(*shape) * 2 * np.pi
        
        return (magnitude * np.exp(1j * phase)).astype(np.complex64)
    
    @staticmethod
    def generate_transducer_positions(
        num_elements: int = 256,
        aperture_size: float = 0.168,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Generate transducer positions."""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate positions in a grid pattern
        grid_size = int(np.sqrt(num_elements))
        if grid_size * grid_size != num_elements:
            # If not perfect square, generate random positions
            return np.random.rand(num_elements, 3) * aperture_size
        
        x = np.linspace(-aperture_size/2, aperture_size/2, grid_size)
        y = np.linspace(-aperture_size/2, aperture_size/2, grid_size)
        X, Y = np.meshgrid(x, y)
        
        positions = np.zeros((num_elements, 3))
        positions[:, 0] = X.flatten()
        positions[:, 1] = Y.flatten()
        positions[:, 2] = 0  # All at z=0
        
        return positions
    
    @staticmethod
    def generate_particle_properties(
        particle_type: str = "polystyrene"
    ) -> Dict[str, Any]:
        """Generate particle properties for testing."""
        properties = {
            "polystyrene": {
                "radius": 1.5e-3,  # m
                "density": 25.0,   # kg/m³
                "acoustic_contrast": 0.3,
                "speed_of_sound": 2350  # m/s
            },
            "water_droplet": {
                "radius": 1.0e-3,  # m
                "density": 1000.0, # kg/m³
                "acoustic_contrast": -0.4,
                "speed_of_sound": 1482  # m/s
            },
            "air_bubble": {
                "radius": 0.5e-3, # m
                "density": 1.225,  # kg/m³
                "acoustic_contrast": -1.0,
                "speed_of_sound": 343  # m/s
            }
        }
        
        return properties.get(particle_type, properties["polystyrene"])


class MockHardware:
    """Mock hardware components for testing."""
    
    @staticmethod
    def create_mock_transducer_array(
        elements: int = 256,
        frequency: float = 40000,
        array_type: str = "mock_array"
    ) -> Mock:
        """Create mock transducer array."""
        transducer = Mock()
        transducer.elements = elements
        transducer.frequency = frequency
        transducer.type = array_type
        transducer.serial_number = f"MOCK-{elements:03d}"
        transducer.positions = MockDataGenerator.generate_transducer_positions(elements)
        
        # Mock methods
        transducer.set_phases = MagicMock()
        transducer.get_phases = MagicMock(
            return_value=np.zeros(elements)
        )
        transducer.set_amplitudes = MagicMock()
        transducer.get_amplitudes = MagicMock(
            return_value=np.ones(elements)
        )
        transducer.activate = MagicMock()
        transducer.deactivate = MagicMock()
        transducer.get_status = MagicMock(return_value="ready")
        transducer.emergency_stop = MagicMock()
        transducer.get_temperature = MagicMock(return_value=25.0)
        transducer.get_power = MagicMock(return_value=10.0)
        
        return transducer
    
    @staticmethod
    def create_mock_hardware_interface(
        device_id: str = "MOCK_HW_001"
    ) -> Mock:
        """Create mock hardware interface."""
        interface = Mock()
        interface.device_id = device_id
        interface.is_connected = False
        
        # Connection methods
        def mock_connect():
            interface.is_connected = True
            return True
        
        def mock_disconnect():
            interface.is_connected = False
            return True
        
        interface.connect = MagicMock(side_effect=mock_connect)
        interface.disconnect = MagicMock(side_effect=mock_disconnect)
        interface.get_status = MagicMock(return_value="connected")
        
        # Data transfer methods
        interface.send_phases = MagicMock()
        interface.send_amplitudes = MagicMock()
        interface.get_feedback = MagicMock(return_value={
            "temperature": 25.0,
            "power": 10.0,
            "voltage": 12.0
        })
        
        # Safety methods
        interface.emergency_stop = MagicMock()
        interface.get_safety_status = MagicMock(return_value={"all_ok": True})
        
        return interface


class TestDataManager:
    """Manage test data files and fixtures."""
    
    def __init__(self):
        self.data_dir = TestConfig.get_test_data_dir()
        self.fixtures_dir = TestConfig.get_fixtures_dir()
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure test directories exist."""
        self.data_dir.mkdir(exist_ok=True)
        self.fixtures_dir.mkdir(exist_ok=True)
    
    def save_test_data(self, name: str, data: Any) -> Path:
        """Save test data to file."""
        file_path = self.data_dir / f"{name}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(data, np.ndarray):
            data = {
                "data": data.tolist(),
                "shape": data.shape,
                "dtype": str(data.dtype)
            }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return file_path
    
    def load_test_data(self, name: str) -> Any:
        """Load test data from file."""
        file_path = self.data_dir / f"{name}.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Test data file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert back to numpy array if it was stored as such
        if isinstance(data, dict) and "data" in data and "shape" in data:
            return np.array(data["data"]).reshape(data["shape"])
        
        return data
    
    def create_reference_data(self):
        """Create reference data for validation tests."""
        references = {
            "single_focus_efficiency": 0.85,
            "twin_trap_isolation": 20.0,
            "phase_stability": 0.02,
            "field_uniformity": 0.1,
            "optimization_convergence": 1e-5,
            "computation_time_limits": {
                "64_elements": 2.0,
                "256_elements": 10.0,
                "1024_elements": 60.0
            },
            "pressure_field_metrics": {
                "max_pressure": 5000,
                "focus_width_fwhm": 0.01,
                "sidelobe_ratio": -20
            }
        }
        
        return self.save_test_data("reference_results", references)


class PerformanceProfiler:
    """Profile test performance and memory usage."""
    
    def __init__(self):
        self.measurements = {}
    
    def start_measurement(self, name: str):
        """Start performance measurement."""
        import time
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        self.measurements[name] = {
            "start_time": time.time(),
            "start_memory": process.memory_info().rss
        }
    
    def end_measurement(self, name: str) -> Dict[str, float]:
        """End performance measurement and return results."""
        import time
        import psutil
        import os
        
        if name not in self.measurements:
            raise ValueError(f"Measurement '{name}' not started")
        
        process = psutil.Process(os.getpid())
        end_time = time.time()
        end_memory = process.memory_info().rss
        
        start_data = self.measurements[name]
        result = {
            "duration": end_time - start_data["start_time"],
            "memory_increase": (end_memory - start_data["start_memory"]) / (1024 * 1024),  # MB
            "peak_memory": end_memory / (1024 * 1024)  # MB
        }
        
        del self.measurements[name]
        return result


class ValidationHelpers:
    """Helper functions for test validation."""
    
    @staticmethod
    def validate_phases(phases: np.ndarray) -> bool:
        """Validate phase array."""
        if not isinstance(phases, np.ndarray):
            return False
        if phases.ndim != 1:
            return False
        if not np.all((phases >= 0) & (phases <= 2 * np.pi)):
            return False
        return True
    
    @staticmethod
    def validate_amplitudes(amplitudes: np.ndarray) -> bool:
        """Validate amplitude array."""
        if not isinstance(amplitudes, np.ndarray):
            return False
        if amplitudes.ndim != 1:
            return False
        if not np.all((amplitudes >= 0) & (amplitudes <= 1.0)):
            return False
        return True
    
    @staticmethod
    def validate_pressure_field(
        field: np.ndarray,
        max_pressure: float = 5000.0
    ) -> bool:
        """Validate pressure field."""
        if not isinstance(field, np.ndarray):
            return False
        if field.ndim != 3:
            return False
        if not np.all(field >= 0):
            return False
        if np.max(field) > max_pressure:
            return False
        return True
    
    @staticmethod
    def validate_optimization_result(result: Dict[str, Any]) -> bool:
        """Validate optimization result structure."""
        required_keys = [
            "phases", "amplitudes", "final_cost", 
            "iterations", "converged"
        ]
        
        if not all(key in result for key in required_keys):
            return False
        
        if not ValidationHelpers.validate_phases(np.array(result["phases"])):
            return False
        
        if not ValidationHelpers.validate_amplitudes(np.array(result["amplitudes"])):
            return False
        
        if not isinstance(result["final_cost"], (int, float)):
            return False
        
        if not isinstance(result["iterations"], int):
            return False
        
        if not isinstance(result["converged"], bool):
            return False
        
        return True
    
    @staticmethod
    def compare_fields(
        field1: np.ndarray,
        field2: np.ndarray,
        tolerance: float = 1e-6
    ) -> bool:
        """Compare two fields within tolerance."""
        if field1.shape != field2.shape:
            return False
        
        return np.allclose(field1, field2, rtol=tolerance, atol=tolerance)
    
    @staticmethod
    def calculate_field_metrics(field: np.ndarray) -> Dict[str, float]:
        """Calculate standard field metrics."""
        return {
            "max_pressure": float(np.max(field)),
            "min_pressure": float(np.min(field)),
            "mean_pressure": float(np.mean(field)),
            "std_pressure": float(np.std(field)),
            "rms_pressure": float(np.sqrt(np.mean(field**2))),
            "energy": float(np.sum(field**2))
        }


# Global test utilities
test_config = TestConfig()
mock_data_generator = MockDataGenerator()
mock_hardware = MockHardware()
test_data_manager = TestDataManager()
performance_profiler = PerformanceProfiler()
validation_helpers = ValidationHelpers()


# Pytest markers for test categorization
def requires_gpu():
    """Mark test as requiring GPU."""
    return pytest.mark.gpu

def requires_hardware():
    """Mark test as requiring hardware."""
    return pytest.mark.hardware

def slow_test():
    """Mark test as slow."""
    return pytest.mark.slow

def integration_test():
    """Mark test as integration test."""
    return pytest.mark.integration

def performance_test():
    """Mark test as performance test."""
    return pytest.mark.performance