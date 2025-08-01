"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock


@pytest.fixture
def mock_transducer():
    """Mock transducer array for testing."""
    transducer = Mock()
    transducer.positions = np.random.rand(256, 3)
    transducer.elements = 256
    transducer.frequency = 40000
    transducer.type = "mock_array"
    transducer.set_phases = MagicMock()
    transducer.activate = MagicMock()
    return transducer


@pytest.fixture
def sample_pressure_field():
    """Sample 3D pressure field for testing."""
    return np.random.rand(100, 100, 100)


@pytest.fixture
def acoustic_hologram_params():
    """Standard parameters for AcousticHologram testing."""
    return {
        "frequency": 40000,
        "medium": "air",
        "optimization_method": "adam",
        "iterations": 100
    }


@pytest.fixture
def mock_hardware_interface():
    """Mock hardware interface for integration testing."""
    interface = Mock()
    interface.connect = MagicMock(return_value=True)
    interface.disconnect = MagicMock()
    interface.send_phases = MagicMock()
    interface.get_status = MagicMock(return_value="connected")
    return interface


@pytest.fixture
def integration_test_config():
    """Configuration for integration tests."""
    return {
        "test_data_dir": "tests/data",
        "reference_results": "tests/references",
        "tolerance": 1e-6,
        "timeout": 30.0
    }


@pytest.fixture(scope="session")
def performance_benchmark_data():
    """Large dataset for performance testing."""
    return {
        "large_array": np.random.rand(1000, 1000, 100),
        "complex_field": np.random.rand(500, 500, 200) + 1j * np.random.rand(500, 500, 200),
        "benchmark_cases": [
            {"size": 128, "frequency": 40000},
            {"size": 256, "frequency": 40000},
            {"size": 512, "frequency": 40000}
        ]
    }