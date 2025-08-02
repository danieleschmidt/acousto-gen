"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
import warnings

# Suppress specific warnings during testing
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "hardware: marks tests that require hardware"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests for performance benchmarking"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "network: marks tests that require network access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    if config.getoption("--runslow"):
        return
    
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_hardware = pytest.mark.skip(reason="hardware tests require --hardware option")
    skip_gpu = pytest.mark.skip(reason="GPU tests require --gpu option")
    
    for item in items:
        if "slow" in item.keywords and not config.getoption("--runslow"):
            item.add_marker(skip_slow)
        if "hardware" in item.keywords and not config.getoption("--hardware"):
            item.add_marker(skip_hardware)
        if "gpu" in item.keywords and not config.getoption("--gpu"):
            item.add_marker(skip_gpu)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--hardware", action="store_true", default=False, help="run hardware tests"
    )
    parser.addoption(
        "--gpu", action="store_true", default=False, help="run GPU tests"
    )
    parser.addoption(
        "--integration", action="store_true", default=False, help="run integration tests"
    )


# ============================================================================
# CORE FIXTURES
# ============================================================================

@pytest.fixture
def mock_transducer():
    """Mock transducer array for testing."""
    transducer = Mock()
    transducer.elements = 256
    transducer.positions = np.random.rand(256, 3) * 0.168  # 16.8cm aperture
    transducer.frequency = 40000
    transducer.type = "mock_array"
    transducer.serial_number = "MOCK-001"
    transducer.calibration_date = "2025-01-01"
    
    # Mock methods
    transducer.set_phases = MagicMock()
    transducer.get_phases = MagicMock(return_value=np.zeros(256))
    transducer.set_amplitudes = MagicMock()
    transducer.get_amplitudes = MagicMock(return_value=np.ones(256))
    transducer.activate = MagicMock()
    transducer.deactivate = MagicMock()
    transducer.get_status = MagicMock(return_value="ready")
    transducer.emergency_stop = MagicMock()
    
    return transducer


@pytest.fixture
def large_mock_transducer():
    """Mock large transducer array for performance testing."""
    transducer = Mock()
    transducer.elements = 1024
    transducer.positions = np.random.rand(1024, 3) * 0.32  # 32cm aperture
    transducer.frequency = 40000
    transducer.type = "large_mock_array"
    
    # Mock methods
    transducer.set_phases = MagicMock()
    transducer.activate = MagicMock()
    transducer.get_status = MagicMock(return_value="ready")
    
    return transducer


@pytest.fixture
def medical_mock_transducer():
    """Mock medical transducer array."""
    transducer = Mock()
    transducer.elements = 256
    transducer.positions = np.random.rand(256, 3) * 0.15  # 15cm aperture
    transducer.frequency = 1500000  # 1.5 MHz
    transducer.type = "medical_array"
    transducer.regulatory_approval = "FDA_Class_II"
    
    # Mock methods with safety features
    transducer.set_phases = MagicMock()
    transducer.activate = MagicMock()
    transducer.emergency_stop = MagicMock()
    transducer.get_temperature = MagicMock(return_value=37.0)  # Body temperature
    transducer.get_power = MagicMock(return_value=50.0)  # Watts
    
    return transducer


# ============================================================================
# DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_pressure_field():
    """Sample 3D pressure field for testing."""
    shape = (100, 100, 50)
    # Create a realistic pressure field with a focus point
    x, y, z = np.meshgrid(
        np.linspace(-0.05, 0.05, shape[0]),
        np.linspace(-0.05, 0.05, shape[1]), 
        np.linspace(0.05, 0.15, shape[2]),
        indexing='ij'
    )
    
    # Gaussian focus at center
    focus_x, focus_y, focus_z = 0, 0, 0.1
    sigma = 0.01  # 1cm width
    
    distance = np.sqrt((x - focus_x)**2 + (y - focus_y)**2 + (z - focus_z)**2)
    pressure = 4000 * np.exp(-(distance / sigma)**2)
    
    return pressure.astype(np.float32)


@pytest.fixture
def complex_pressure_field():
    """Complex pressure field with phase information."""
    shape = (50, 50, 25)
    # Create field with both magnitude and phase
    magnitude = np.random.rand(*shape) * 1000  # 0-1000 Pa
    phase = np.random.rand(*shape) * 2 * np.pi  # 0-2π
    
    complex_field = magnitude * np.exp(1j * phase)
    return complex_field.astype(np.complex64)


@pytest.fixture
def test_configuration():
    """Load test configuration data."""
    config_path = Path(__file__).parent / "fixtures" / "test_data.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    else:
        # Fallback configuration if file doesn't exist
        return {
            "acoustic_parameters": {
                "frequencies": [20000, 40000, 60000],
                "media": {
                    "air": {"density": 1.225, "speed_of_sound": 343.0},
                    "water": {"density": 1000.0, "speed_of_sound": 1482.0}
                }
            },
            "safety_limits": {
                "max_pressure": 5000,
                "max_intensity": 15
            }
        }


@pytest.fixture
def acoustic_hologram_params():
    """Standard parameters for AcousticHologram testing."""
    return {
        "frequency": 40000,
        "medium": "air",
        "optimization_method": "adam",
        "iterations": 100,
        "convergence_threshold": 1e-6,
        "learning_rate": 0.01
    }


# ============================================================================
# HARDWARE FIXTURES
# ============================================================================

@pytest.fixture
def mock_hardware_interface():
    """Mock hardware interface for integration testing."""
    interface = Mock()
    interface.device_id = "MOCK_HW_001"
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
    interface.get_feedback = MagicMock(return_value={"temperature": 25.0, "power": 10.0})
    
    # Safety methods
    interface.emergency_stop = MagicMock()
    interface.get_safety_status = MagicMock(return_value={"all_ok": True})
    
    return interface


@pytest.fixture
def mock_calibration_data():
    """Mock calibration data for testing."""
    num_elements = 256
    
    calibration = {
        "timestamp": "2025-01-01T00:00:00Z",
        "array_serial": "MOCK-001",
        "calibration_version": "1.0",
        "phase_corrections": np.random.normal(0, 0.1, num_elements),
        "amplitude_corrections": np.random.normal(1.0, 0.05, num_elements),
        "frequency_response": np.ones(num_elements),
        "position_corrections": np.random.normal(0, 0.001, (num_elements, 3)),
        "element_status": np.ones(num_elements, dtype=bool),  # All elements working
        "calibration_quality": 0.95,  # 95% quality score
        "environmental_conditions": {
            "temperature": 22.0,  # °C
            "humidity": 45.0,     # %
            "pressure": 101325.0  # Pa
        }
    }
    
    return calibration


# ============================================================================
# TEST ENVIRONMENT FIXTURES
# ============================================================================

@pytest.fixture
def temp_directory():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_data_directory():
    """Test data directory."""
    test_dir = Path(__file__).parent / "data"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture
def integration_test_config():
    """Configuration for integration tests."""
    return {
        "test_data_dir": "tests/data",
        "reference_results": "tests/references", 
        "tolerance": 1e-6,
        "timeout": 30.0,
        "max_iterations": 1000,
        "convergence_patience": 100
    }


# ============================================================================
# PERFORMANCE FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def performance_benchmark_data():
    """Large dataset for performance testing."""
    return {
        "small_field": np.random.rand(64, 64, 32).astype(np.float32),
        "medium_field": np.random.rand(128, 128, 64).astype(np.float32),
        "large_field": np.random.rand(256, 256, 128).astype(np.float32),
        "complex_field": (
            np.random.rand(100, 100, 50).astype(np.float32) +
            1j * np.random.rand(100, 100, 50).astype(np.float32)
        ),
        "benchmark_cases": [
            {"elements": 64, "frequency": 40000, "expected_time": 1.0},
            {"elements": 256, "frequency": 40000, "expected_time": 5.0},
            {"elements": 1024, "frequency": 40000, "expected_time": 20.0}
        ]
    }


@pytest.fixture
def memory_monitor():
    """Memory usage monitoring fixture."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    class MemoryMonitor:
        def __init__(self):
            self.initial_memory = initial_memory
            self.process = process
            
        def current_usage(self):
            return self.process.memory_info().rss
            
        def memory_increase(self):
            return self.current_usage() - self.initial_memory
            
        def memory_increase_mb(self):
            return self.memory_increase() / (1024 * 1024)
    
    return MemoryMonitor()


# ============================================================================
# SAFETY AND VALIDATION FIXTURES
# ============================================================================

@pytest.fixture
def safety_validator():
    """Safety validation fixture."""
    class SafetyValidator:
        def __init__(self):
            self.max_pressure = 5000  # Pa
            self.max_intensity = 15   # W/cm²
            self.max_temperature = 45 # °C
            
        def validate_pressure(self, pressure):
            return np.all(pressure <= self.max_pressure)
            
        def validate_intensity(self, intensity):
            return np.all(intensity <= self.max_intensity)
            
        def validate_temperature(self, temperature):
            return temperature <= self.max_temperature
            
        def validate_all(self, pressure, intensity, temperature):
            return (
                self.validate_pressure(pressure) and
                self.validate_intensity(intensity) and 
                self.validate_temperature(temperature)
            )
    
    return SafetyValidator()


@pytest.fixture
def reference_results():
    """Reference results for validation testing."""
    return {
        "single_focus_efficiency": 0.85,  # Expected focusing efficiency
        "twin_trap_isolation": 20.0,     # dB isolation between traps
        "phase_stability": 0.02,         # RMS phase variation
        "field_uniformity": 0.1,         # RMS field uniformity
        "optimization_convergence": 1e-5, # Expected convergence
        "computation_time_limits": {
            "64_elements": 2.0,   # seconds
            "256_elements": 10.0, # seconds
            "1024_elements": 60.0 # seconds
        }
    }


# ============================================================================
# APPLICATION-SPECIFIC FIXTURES
# ============================================================================

@pytest.fixture
def levitation_test_config():
    """Configuration for levitation testing."""
    return {
        "particle_properties": {
            "polystyrene_3mm": {
                "radius": 1.5e-3,      # m
                "density": 25.0,       # kg/m³
                "acoustic_contrast": 0.3
            },
            "water_droplet_2mm": {
                "radius": 1.0e-3,      # m
                "density": 1000.0,     # kg/m³
                "acoustic_contrast": -0.4
            }
        },
        "trap_parameters": {
            "minimum_trap_strength": 2000,  # Pa
            "optimal_trap_height": 0.08,    # m
            "stability_threshold": 0.1      # Relative stability
        }
    }


@pytest.fixture
def haptic_test_config():
    """Configuration for haptic testing."""
    return {
        "perception_thresholds": {
            "detection_threshold": 50,   # Pa
            "comfortable_level": 150,    # Pa
            "discomfort_threshold": 300  # Pa
        },
        "modulation_parameters": {
            "optimal_frequency": 200,    # Hz
            "frequency_range": [20, 400], # Hz
            "duty_cycle": 0.5           # 50%
        },
        "spatial_parameters": {
            "minimum_resolution": 0.01,  # m
            "working_distance": 0.15,    # m
            "maximum_force": 0.001       # N
        }
    }


@pytest.fixture
def medical_test_config():
    """Configuration for medical application testing."""
    return {
        "regulatory_limits": {
            "fda_max_pressure": 1000000,  # Pa (1 MPa)
            "iec_max_intensity": 720,     # W/cm²
            "thermal_dose_limit": 240     # CEM43
        },
        "tissue_properties": {
            "muscle": {
                "density": 1050,        # kg/m³
                "speed_of_sound": 1540, # m/s
                "absorption": 0.05,     # Np/m/MHz
                "thermal_conductivity": 0.5 # W/m/K
            },
            "fat": {
                "density": 920,         # kg/m³
                "speed_of_sound": 1450, # m/s
                "absorption": 0.06,     # Np/m/MHz
                "thermal_conductivity": 0.2 # W/m/K
            }
        },
        "treatment_parameters": {
            "focus_precision": 0.002,   # m (2mm)
            "heating_rate": 2.0,        # °C/s
            "cooling_time": 30.0        # s
        }
    }